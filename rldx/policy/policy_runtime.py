# SPDX-License-Identifier: Apache-2.0
"""
PolicyRuntime — inference pipeline orchestration, extracted from
RLDXPolicy._get_action.

Before this extraction, the full pipeline (unbatch → process → collate →
RTC prefix → memory scratchpad + model.get_action → RTC save → decode)
lived as a single ~150 LOC method inside the RLDXPolicy god-class. Problems:
  - Pipeline stages were implicit (code order only) — hard to see the
    sequence at a glance
  - Each stage was inline, not unit-testable
  - RLDXPolicy held every dependency (model, processor, registry,
    infer_lock, rtc config) — no way to swap for testing

PolicyRuntime owns the inference pipeline as a set of named stages:
  step()                → top-level orchestration
  _prepare_inputs()     → unbatch + VLA conversion + processor + collate
  _inject_rtc_prefix()  → RTC cache invalidate + prefix build (client or
                          server cache) + inject into collated
  _inject_eag_guide()   → SAIL EAG guide (physical units, from the client)
                          normalized + injected as ``eag_guide_actions``
  _run_inference()      → memory scratchpad wrap + autocast + model.get_action
  _decode()             → normalized_action → physical action dict

Dependencies are injected via __init__ — Runtime doesn't own loading or
validation, just the inference sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import MessageType, VLAStepData

from .session_registry import SessionRegistry
from .step_request import StepRequest


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert floating-point tensors in a nested structure.

    Non-floating-point tensors are left as-is. Handles dict-like objects
    (BatchFeature has items() but is not a dict).
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


class PolicyRuntime:
    """Orchestrates the inference pipeline for RLDXPolicy.

    All state lives elsewhere:
      - model / processor: owned by caller (RLDXPolicy), here via DI
      - session state: owned by SessionRegistry
      - config: resolved at load time, injected as primitives

    Runtime is functional-ish — step() is the entry point and sub-methods
    are pipeline stages. Sub-methods mutate the collated dict in-place;
    this is a known tradeoff vs fully functional (threading full state
    through return values would be verbose).
    """

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        modality_configs: dict,
        embodiment_tag: EmbodimentTag,
        collate_fn: Any,
        language_key: str,
        registry: SessionRegistry,
        infer_lock,
        use_memory: bool,
        rtc_inference_mode: str,
        rtc_inference_delay: int,
        rtc_exec_horizon: int,
        rtc_enabled: bool,
        verbose: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.modality_configs = modality_configs
        self.embodiment_tag = embodiment_tag
        self.collate_fn = collate_fn
        self.language_key = language_key
        self.registry = registry
        self._infer_lock = infer_lock
        self.use_memory = use_memory
        self.rtc_inference_mode = rtc_inference_mode
        self.rtc_inference_delay = rtc_inference_delay
        self.rtc_exec_horizon = rtc_exec_horizon
        self._rtc_enabled = rtc_enabled
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def step(self, request: StepRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        """Full inference pipeline.

        Stages:
          1. Prepare inputs   (unbatch → VLA → processor → collate)
          2. RTC prefix inject (if _rtc_enabled)
          3. Inference         (memory scratchpad if use_memory, else bare)
          4. RTC chunk save    (if _rtc_enabled)
          5. Decode            (normalized → physical action dict)
        """
        # Build reset_memory tensor from StepRequest.reset_mask
        reset_memory = (
            torch.tensor(
                request.reset_mask,
                device=self.model.device,
                dtype=torch.bool,
            )
            if request.reset_mask is not None
            else None
        )
        if self.verbose:
            if reset_memory is not None and reset_memory.any():
                print(
                    f"[SERVER-LOG] Received reset_memory={request.reset_mask} "
                    f"for session_ids={request.sids}"
                )

        unbatched_obs, states, collated = self._prepare_inputs(request)
        B = len(unbatched_obs)

        # Inject reset_memory for memory-enabled RLDX (pre-RTC so RTC block
        # can override the tensor's values via cold_start mask later).
        if self.use_memory and reset_memory is not None:
            collated["reset_memory"] = reset_memory

        active_sids = self._inject_rtc_prefix(request, collated, B, reset_memory)
        self._inject_eag_guide(request, collated, B)

        model_pred, reset_memory = self._run_inference(
            request,
            collated,
            B,
            reset_memory,
        )

        normalized_action = model_pred["action_pred"].float()

        # RTC chunk cache update with freshly predicted chunk
        if self._rtc_enabled and active_sids:
            self.registry.save_rtc_batch(active_sids, normalized_action)

        action = self._decode(normalized_action, states)
        return action, {}

    # ------------------------------------------------------------------
    # Stage 1: prepare inputs
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        request: StepRequest,
    ) -> tuple[list[dict], list[dict], dict]:
        """Unbatch → VLAStepData → processor → collate → bfloat16.

        Returns (unbatched_obs, states, collated_inputs). states is kept
        separately for later decode step (per-sample state arrays).
        """
        unbatched_obs = self._unbatch_observation(request.obs)
        processed_inputs = []
        states: list[dict] = []
        for obs in unbatched_obs:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated = self.collate_fn(processed_inputs)

        if self.verbose:
            print(f"[SERVER-LOG] Collated inputs keys: {list(collated.keys())}")
            for modality in ("video", "state"):
                if modality in collated:
                    for k, v in collated[modality].items():
                        print(f"[SERVER-LOG] collated['{modality}'][{k}] shape: {v.shape}")

        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        return unbatched_obs, states, collated

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Split batched observation into a list of single observations."""
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]
        unbatched_obs = []
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            if "physics" in value:
                unbatched_value["physics"] = {k: v[i] for k, v in value["physics"].items()}
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
            physics=observation.get("physics", {}),
        )

    # ------------------------------------------------------------------
    # Stage 2: RTC prefix injection
    # ------------------------------------------------------------------

    def _inject_rtc_prefix(
        self,
        request: StepRequest,
        collated: dict,
        B: int,
        reset_memory: "torch.Tensor | None",
    ) -> list[str]:
        """Build + inject RTC action prefix into collated inputs.

        Priority: client-supplied prefix > server cache fallback > cold start.
        Returns active_sids (empty list if RTC disabled) — caller uses it
        for post-inference chunk save.

        No-op if _rtc_enabled is False.
        """
        if not self._rtc_enabled:
            return []

        active_sids = self.registry.resolve_sids(request.sids, B)
        # RTC block only invalidates the chunk cache. Session-level
        # termination is the memory block's job.
        self.registry.invalidate_rtc(active_sids, reset_memory)

        d, s = self.rtc_inference_delay, self.rtc_exec_horizon

        # Path 1: client-supplied prefix (authoritative — real robots
        # measure true inference latency, more accurate than cache)
        client_prefix = request.action_prefix
        client_prefix_len = request.rtc_prefix_len

        prefix_stack = None
        effective_len = None
        prefix_source = None
        if client_prefix is not None:
            if not isinstance(client_prefix, torch.Tensor):
                client_prefix = torch.as_tensor(client_prefix, dtype=torch.float32)
            if client_prefix.dim() == 2:
                client_prefix = client_prefix.unsqueeze(0).expand(B, -1, -1)
            if client_prefix.shape[0] != B:
                raise ValueError(
                    f"client-supplied action_prefix has batch="
                    f"{client_prefix.shape[0]} but request B={B}"
                )
            effective_len = int(client_prefix_len) if client_prefix_len is not None else d
            prefix_stack = client_prefix[:, :effective_len].contiguous()

            # Client sends decoded (physical-unit) actions back as prefix.
            # Normalize them into the model's action space before injection;
            # server-cache path already stores normalized actions.
            prefix_stack = self._normalize_and_pad(prefix_stack, "action_prefix")
            prefix_source = "client"
        else:
            # Path 2: server cache fallback
            prefix_stack = self.registry.load_rtc_prefix(active_sids, d, s)
            if prefix_stack is not None:
                effective_len = d
                prefix_source = "server_cache"

        if prefix_stack is not None:
            collated["action_prefix"] = prefix_stack.to(
                self.model.device,
                dtype=torch.bfloat16,
            )
            collated["rtc_prefix_len"] = effective_len
            if self.verbose:
                print(
                    f"[SERVER-LOG] RTC prefix injected: source={prefix_source} "
                    f"B={B} d={effective_len} s={s}"
                )
        elif self.verbose:
            print("[SERVER-LOG] RTC prefix: cold start (no client prefix, no server cache)")

        # Postfix target Y[d:] for guided-mode Jacobian VJP (Eq. 5 of arXiv
        # 2506.07339). Source is always the server cache: ``save_rtc_batch``
        # populated it with the previous chunk's full normalized action plan.
        # Trained mode and cold starts skip this — Y collapses to its
        # attention-coupling fallback inside the model.
        if self.rtc_inference_mode == "guided" and prefix_stack is not None:
            postfix_target = self.registry.load_rtc_postfix_target(active_sids, d, s)
            if postfix_target is not None:
                collated["action_postfix_target"] = postfix_target.to(
                    self.model.device,
                    dtype=torch.bfloat16,
                )
                if self.verbose:
                    print(
                        f"[SERVER-LOG] RTC postfix Y target loaded: "
                        f"shape={tuple(postfix_target.shape)}"
                    )
            elif self.verbose:
                print("[SERVER-LOG] RTC postfix Y target: cache miss (cold start ramp)")
        return active_sids

    # ------------------------------------------------------------------
    # Physical → normalized action conversion (shared by RTC prefix and EAG)
    # ------------------------------------------------------------------

    def _normalize_and_pad(self, actions: "torch.Tensor", what: str) -> "torch.Tensor":
        """Normalize physical-unit actions (B, T, D) and zero-pad to action_dim.

        The client speaks physical units — the same units ``_decode`` hands
        back — so anything it sends into the model's action space (an RTC
        prefix, a SAIL EAG guide) goes through the processor's own
        ``apply_action`` first. Padding takes real_dim → the model's
        ``action_dim`` (e.g. ALLEX 48 → 64), which is the width the action
        encoders were built for.
        """
        sap = self.processor.state_action_processor
        tag = (
            self.embodiment_tag.value
            if hasattr(self.embodiment_tag, "value")
            else str(self.embodiment_tag)
        )
        modality_keys = sap.modality_configs[tag]["action"].modality_keys
        joint_dims = []
        for k in modality_keys:
            p = sap.norm_params[tag]["action"][k]
            first = next(arr_k for arr_k in ("min", "mean", "q01") if arr_k in p)
            joint_dims.append(np.asarray(p[first]).shape[0])
        real_dim = sum(joint_dims)
        arr_np = actions[:, :, :real_dim].float().cpu().numpy()  # (B, T, real_dim)
        B_, T_, _ = arr_np.shape
        normalized = np.zeros_like(arr_np)
        for b in range(B_):
            cursor = 0
            action_dict = {}
            for k, w in zip(modality_keys, joint_dims):
                action_dict[k] = arr_np[b, :, cursor : cursor + w]
                cursor += w
            norm_dict = sap.apply_action(action_dict, embodiment_tag=tag, state=None)
            cursor = 0
            for k, w in zip(modality_keys, joint_dims):
                normalized[b, :, cursor : cursor + w] = norm_dict[k]
                cursor += w
        pad = actions.shape[-1] - real_dim
        if pad > 0:
            normalized = np.concatenate(
                [normalized, np.zeros((B_, T_, pad), dtype=normalized.dtype)], axis=-1
            )
        out = torch.as_tensor(normalized, dtype=actions.dtype, device=actions.device)

        expected_dim = self.model.action_model.action_dim
        actual_dim = out.shape[-1]
        if actual_dim < expected_dim:
            out = torch.nn.functional.pad(out, (0, expected_dim - actual_dim))
        elif actual_dim > expected_dim:
            raise ValueError(
                f"client {what} dim {actual_dim} exceeds model "
                f"max_action_dim {expected_dim} — caller is sending more "
                f"channels than the model can consume"
            )
        return out

    # ------------------------------------------------------------------
    # Stage 2b: SAIL EAG guide injection
    # ------------------------------------------------------------------

    def _inject_eag_guide(self, request: StepRequest, collated: dict, B: int) -> None:
        """Normalize + inject the client's EAG guide as ``eag_guide_actions``.

        SAIL's Error-Adaptive Guidance conditions the chunk on the tail of the
        previous chunk that has not been executed yet. The client owns that
        tail (and the tracking-error gate that decides whether to send it at
        all), so the guide arrives per request in ``options["eag_guide"]``,
        in physical units, shaped (h, D) or (B, h, D).

        No guide in the request means the unconditional branch alone runs —
        which is what the model does at episode start, and what a gate that
        stays closed asks for. A guide on a checkpoint that was not trained
        with EAG is a caller error, not something to drop silently.
        """
        guide = request.eag_guide
        if guide is None:
            return
        if not getattr(self.model.action_model, "use_future_action_condition", False):
            raise ValueError(
                "options['eag_guide'] was sent to a checkpoint trained without "
                "EAG (use_future_action_condition=False)"
            )
        if not isinstance(guide, torch.Tensor):
            guide = torch.as_tensor(np.asarray(guide), dtype=torch.float32)
        guide = guide.float()
        if guide.dim() == 2:
            guide = guide.unsqueeze(0).expand(B, -1, -1)
        if guide.shape[0] != B:
            raise ValueError(
                f"client-supplied eag_guide has batch={guide.shape[0]} but request B={B}"
            )
        trained_h = int(self.model.action_model.future_action_condition_horizon)
        if guide.shape[1] > trained_h:
            guide = guide[:, :trained_h]
        guide = self._normalize_and_pad(guide.contiguous(), "eag_guide")
        collated["eag_guide_actions"] = guide.to(self.model.device, dtype=torch.bfloat16)
        if self.verbose:
            print(
                f"[SERVER-LOG] EAG guide injected: B={B} h={guide.shape[1]} "
                f"w={self.model.action_model.eag_cfg_weight}"
            )

    # ------------------------------------------------------------------
    # Stage 3: run inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        request: StepRequest,
        collated: dict,
        B: int,
        reset_memory: "torch.Tensor | None",
    ) -> tuple[dict, "torch.Tensor | None"]:
        """Memory scratchpad (if use_memory) + model.get_action.

        Returns (model_pred, updated_reset_memory). reset_memory may be
        updated to reflect cold-start slots (multi-session memory path).
        """
        if self.use_memory and hasattr(self.model, "_cached_mq"):
            # Registry owns model._cached_mq lifecycle via context manager.
            # Runtime never touches _cached_mq directly.
            with (
                self._infer_lock,
                self.registry.memory_scratchpad(
                    self.model,
                    request.sids,
                    B,
                    reset_memory,
                ) as cold_start,
            ):
                if cold_start is not None:
                    # Multi-session path: mark cold-start slots in
                    # reset_memory tensor so model knows not to read placeholder
                    if reset_memory is None:
                        reset_memory = torch.zeros(
                            B,
                            device=self.model.device,
                            dtype=torch.bool,
                        )
                    for idx, cs in enumerate(cold_start):
                        if cs:
                            reset_memory[idx] = True
                    collated["reset_memory"] = reset_memory

                model_pred = self._forward(collated)
        else:
            model_pred = self._forward(collated)

        return model_pred, reset_memory

    def _forward(self, collated: dict) -> dict:
        """Invoke model.get_action under appropriate autograd + autocast."""
        with (
            self._select_inference_context(
                self._rtc_enabled,
                self.rtc_inference_mode,
            ),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            return self.model.get_action(**collated)

    @staticmethod
    def _select_inference_context(rtc_enabled: bool, rtc_mode: str):
        """Pick gradient context for model.get_action.

        ``torch.inference_mode()`` is default — faster than ``no_grad``
        and rules out gradient bookkeeping. But inference tensors never
        get grad_fn attached even under ``enable_grad``. Guided RTC
        needs a real autograd graph (``torch.autograd.grad``), so that
        path falls back to ``no_grad`` which still skips non-Jacobian
        compute but lets ``enable_grad`` materialise the VJP graph.
        """
        if rtc_enabled and rtc_mode == "guided":
            return torch.no_grad()
        return torch.inference_mode()

    # ------------------------------------------------------------------
    # Stage 4: decode
    # ------------------------------------------------------------------

    def _decode(
        self,
        normalized_action: "torch.Tensor",
        states: list[dict],
    ) -> dict[str, np.ndarray]:
        """Normalized action → physical action dict (float32).

        Per-sample states are stacked along batch dim and passed to the
        processor for denormalization (processor needs the current state
        to invert state-relative actions).
        """
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(),
            self.embodiment_tag,
            batched_states,
        )
        return {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
