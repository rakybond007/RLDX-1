# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

from dataclasses import dataclass
import json
import os
from typing import Literal

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.replay_policy import ReplayPolicy
from rldx.policy.rldx_policy import RLDXPolicy
from rldx.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the RLDX-1 inference server."""

    # RLDX policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    sample_timestep_from_beta_dist: bool = False
    """Whether to sample timestep from beta distribution. If False, sample uniformly."""

    denoising_timesteps: list[float] | None = None
    """Fixed denoising timesteps for inference (e.g., [0.0, 0.1, 0.3, 0.6]).
    If provided, overrides uniform spacing and sample_timestep_from_beta_dist."""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    transport: Literal["zmq", "websocket"] = "zmq"
    """Wire protocol the server speaks.

    ``zmq`` is RLDX's own ZeroMQ + msgpack ``PolicyServer``. ``websocket`` is
    the openpi-compatible server, for clients whose environment carries
    ``websockets`` + ``openpi_client`` but not ``pyzmq`` (the LIBERO eval
    environment is one). Both accept the same request shape, so the same
    client adapter drives either.
    """

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    verbose: bool = False
    """Whether to print verbose debug logs during inference"""

    deactivate_memory: bool = False
    """If True, load a memory-trained model as a non-memory model (RLDX),
    skipping memory module weights. Useful for evaluating memory-dropout models
    without memory at inference time."""

    # ── Real-Time Chunking overrides ───────────────────────────────────────
    rtc_inference_mode: str | None = None
    """Override model.config.rtc_inference_mode at load time.
    One of 'none' / 'trained' / 'guided'. None → use checkpoint's value."""

    rtc_inference_delay: int | None = None
    """Override model.config.rtc_inference_delay (frozen prefix length d)."""

    rtc_inference_exec_horizon: int | None = None
    """Override model.config.rtc_inference_exec_horizon (execution horizon s).
    None resolves to action_horizon - d at policy init."""

    rtc_jacobian_beta: float | None = None
    """Override guidance-scale clip for 'guided' mode."""

    rtc_jacobian_steps_only: int | None = None
    """If set, guide only the first N denoising steps (memory-saving)."""

    # ── SAIL Error-Adaptive Guidance ───────────────────────────────────────
    eag_cfg_weight: float | None = None
    """Override the checkpoint's ``eag_cfg_weight`` at load time.

    The combination is ``uncond + (1 + w)(cond - uncond)``, so the code weight
    and the SAIL paper's weight differ by one: ``paper w = code w + 1``. Code
    ``0.0`` is paper ``w`` 1 (the answer is the conditional branch itself) and
    code ``-1.0`` is paper ``w`` 0 (the answer is the unconditional branch).
    Negative values are passed through. None keeps the checkpoint's value.

    Only meaningful on a checkpoint trained with EAG; a checkpoint without it
    is rejected rather than served with the flag silently dropped.
    """

    # ── Inference acceleration ─────────────────────────────────────────────
    compile: Literal["none", "submodule", "fullgraph"] | None = None
    """Compilation level applied after the policy is loaded.

      none       — eager (default; equivalent to leaving this flag unset)
      submodule  — ``torch.compile`` per learnable sub-module; preserves
                   autograd, so RTC ``guided`` mode works
      fullgraph  — ``torch.compile(fullgraph=True)`` over the whole VLA
                   chain; faster steady-state latency but cannot route
                   VJP, so RTC ``guided`` mode is rejected at startup
    """


_COMPILE_LEVEL_TO_PATH = {"none": "A", "submodule": "B", "fullgraph": "D"}


def _resolve_compile_level(config: ServerConfig) -> str | None:
    """Return the internal optimization path letter (B/D), or None for eager."""
    if config.compile is None:
        return None
    path = _COMPILE_LEVEL_TO_PATH[config.compile]
    return None if path == "A" else path


def _validate_cli(config: ServerConfig, opt_path: str | None) -> None:
    """Reject incompatible flag combinations before we pay the model-load cost
    (~3 min). Defense-in-depth: ``apply_optimization`` re-checks this against
    the loaded checkpoint's actual ``rtc_inference_mode``.
    """
    if opt_path in {"C", "D"} and config.rtc_inference_mode == "guided":
        raise ValueError(
            f"--compile={config.compile or '<via path>'} is incompatible with "
            "--rtc-inference-mode=guided: the compiled fullgraph cannot route "
            "VJP through ``action_prefix``. Use --compile=submodule (path B) "
            "or --rtc-inference-mode in {none, trained}."
        )


def main(config: ServerConfig):
    print("Starting RLDX-1 inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    opt_path = _resolve_compile_level(config)
    _validate_cli(config, opt_path)

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = RLDXPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            sample_timestep_from_beta_dist=config.sample_timestep_from_beta_dist,
            denoising_timesteps=config.denoising_timesteps,
            verbose=config.verbose,
            deactivate_memory=config.deactivate_memory,
            rtc_inference_mode=config.rtc_inference_mode,
            rtc_inference_delay=config.rtc_inference_delay,
            rtc_inference_exec_horizon=config.rtc_inference_exec_horizon,
            rtc_jacobian_beta=config.rtc_jacobian_beta,
            rtc_jacobian_steps_only=config.rtc_jacobian_steps_only,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from rldx.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # EAG: the checkpoint records the weight training used (1.0 by default),
    # and evaluation moves it. A silently ignored override would serve the
    # wrong arm under the right job name, so this prints what it changed and
    # the eval scripts grep the line before launching any client.
    if config.eag_cfg_weight is not None:
        action_model = getattr(policy, "model", None)
        action_model = getattr(action_model, "action_model", None)
        if not getattr(action_model, "use_future_action_condition", False):
            raise SystemExit(
                "--eag-cfg-weight only means something on a checkpoint trained "
                "with EAG (use_future_action_condition=False on this one)"
            )
        prev = action_model.eag_cfg_weight
        action_model.eag_cfg_weight = float(config.eag_cfg_weight)
        policy.model.config.eag_cfg_weight = float(config.eag_cfg_weight)
        print(
            f"  [eag] cfg weight {prev} -> {config.eag_cfg_weight} "
            f"(combination = uncond + (1+w)(cond - uncond), paper w = code w + 1)",
            flush=True,
        )

    # Apply inference optimization if requested. ReplayPolicy has no
    # ``model.get_action`` attribute to wrap, so optimization only fires for
    # the RLDXPolicy branch.
    if opt_path is not None and config.model_path is not None:
        from rldx.inference.serve_optimization import apply_optimization

        info = apply_optimization(policy, path=opt_path)
        print(f"  Inference optimization: path={opt_path}, info={info}")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from rldx.policy.rldx_policy import RLDXSimPolicyWrapper

        policy = RLDXSimPolicyWrapper(policy, strict=config.strict)

    if config.transport == "websocket":
        from rldx.eval.serving import websocket_policy_server

        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0" if config.host in ("127.0.0.1", "*") else config.host,
            port=config.port,
            metadata=None,
        )
        print(f"Server is ready and listening on ws://{config.host}:{config.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        return

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
