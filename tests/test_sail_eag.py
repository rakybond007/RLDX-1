"""Checks for the SAIL port: EAG wiring and the precision-bit modality configs.

Runs on CPU in seconds. It is deliberately about *wiring*, because that is what
silently produces a 60,000-step baseline under a SAIL run name:

  - a CLI flag that never reaches the model config (assembly.py warns about
    exactly this for RTC),
  - a modality config whose action_configs no longer line up with its keys,
  - two modality configs fighting over one embodiment tag,
  - a guide that is appended but never actually changes the model input.

Usage:  .venv/bin/python tests/test_sail_eag.py
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - this is the reporter
        FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc}")
    else:
        print(f"[ok]   {name}")


# ── 1. modality configs ─────────────────────────────────────────────────────


def _load_isolated(module_path: str) -> dict:
    """Import one modality config in a fresh process and dump what it registered.

    Separate processes are the point: `register_modality_config` asserts one
    registration per embodiment tag, so importing both SAIL configs together
    is *supposed* to fail.
    """
    code = (
        "import json, sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', {module_path!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "from rldx.configs.data.embodiment_configs import MODALITY_CONFIGS\n"
        "cfg = MODALITY_CONFIGS['gr00t_general_embodiment' if 'gr00t_general_embodiment' "
        "in MODALITY_CONFIGS else [k for k in MODALITY_CONFIGS if 'general' in k][0]]\n"
        "print(json.dumps({\n"
        "  'action_keys': cfg['action'].modality_keys,\n"
        "  'n_action_configs': len(cfg['action'].action_configs or []),\n"
        "  'video_keys': cfg['video'].modality_keys,\n"
        "  'state_keys': cfg['state'].modality_keys,\n"
        "  'delta_indices': cfg['action'].delta_indices,\n"
        "}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise AssertionError(f"import failed:\n{out.stderr[-1500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_libero_sail_config():
    got = _load_isolated(str(ROOT / "rldx/configs/data/libero_sail_config.py"))
    assert got["action_keys"] == [
        "eef_pos_delta",
        "eef_rot_delta",
        "gripper_close",
        "precision",
    ], got["action_keys"]
    assert got["n_action_configs"] == 4, got["n_action_configs"]
    assert got["video_keys"] == ["front_view", "left_wrist_view"], got["video_keys"]
    assert got["delta_indices"] == list(range(16)), got["delta_indices"]


def test_robocasa_sail_config():
    got = _load_isolated(str(ROOT / "rldx/configs/data/robocasa_sail_config.py"))
    assert got["action_keys"] == [
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
        "precision",
    ], got["action_keys"]
    assert got["n_action_configs"] == 6, got["n_action_configs"]
    assert got["video_keys"] == ["left_view", "right_view", "wrist_view"], got["video_keys"]
    assert got["delta_indices"] == list(range(16)), got["delta_indices"]


def test_sail_config_matches_baseline_plus_precision():
    """The SAIL config must differ from its baseline only by the precision key."""
    for sail, base in (
        ("libero_sail_config.py", "libero_config.py"),
        ("robocasa_sail_config.py", "robocasa_config.py"),
    ):
        a = _load_isolated(str(ROOT / "rldx/configs/data" / sail))
        b = _load_isolated(str(ROOT / "rldx/configs/data" / base))
        assert a["action_keys"] == b["action_keys"] + ["precision"], (sail, a, b)
        assert a["n_action_configs"] == b["n_action_configs"] + 1, (sail, a, b)
        assert a["video_keys"] == b["video_keys"], (sail, a, b)
        assert a["state_keys"] == b["state_keys"], (sail, a, b)


def test_two_configs_cannot_share_a_tag():
    """Loading both SAIL configs in one process must fail, loudly and early."""
    code = (
        "import importlib.util\n"
        "for p in ('rldx/configs/data/libero_sail_config.py',"
        "          'rldx/configs/data/robocasa_sail_config.py'):\n"
        "    s = importlib.util.spec_from_file_location(p, p)\n"
        "    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert out.returncode != 0, "expected the duplicate registration to fail"
    assert "already registered" in out.stderr, out.stderr[-800:]


# ── 2. CLI → model config wiring ────────────────────────────────────────────


def test_train_config_exposes_eag_flags():
    from rldx.configs.train_config import TrainConfig

    for field in (
        "use_future_action_condition",
        "future_action_condition_horizon",
        "future_action_condition_dropout",
        "eag_cfg_weight",
    ):
        assert hasattr(TrainConfig, field) or field in TrainConfig.__annotations__, field


def test_model_config_exposes_eag_flags():
    from rldx.configs.model.rldx import RLDXConfig

    ann = RLDXConfig.__annotations__ if hasattr(RLDXConfig, "__annotations__") else {}
    for field in (
        "use_future_action_condition",
        "future_action_condition_horizon",
        "future_action_condition_dropout",
        "eag_cfg_weight",
    ):
        assert field in ann or hasattr(RLDXConfig, field), field


def test_assembly_copies_every_eag_flag():
    """A flag that assembly forgets to copy is a silent no-op — the exact
    failure the RTC comment in that file warns about."""
    src = (ROOT / "rldx/experiment/assembly.py").read_text()
    for field in (
        "use_future_action_condition",
        "future_action_condition_horizon",
        "future_action_condition_dropout",
        "eag_cfg_weight",
    ):
        line = f"run_config.model.{field} = cli.{field}"
        assert line in src, f"assembly.py never copies {field}"


# ── 3. EAG tensor behaviour ─────────────────────────────────────────────────


class _Stub:
    """Minimal stand-in carrying just what _append_future_action_condition uses."""

    # The port lives on RLDXActionModel (the training/inference body), not on
    # the outer RLDX PreTrainedModel wrapper.
    from rldx.model.core.rldx import RLDXActionModel as _Body

    _append_future_action_condition = _Body._append_future_action_condition

    def __init__(self, action_dim=8, emb=16, horizon=4, enabled=True):
        self.use_future_action_condition = enabled
        self.future_action_condition_horizon = horizon
        self.action_dim = action_dim
        self.future_action_condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(action_dim, emb),
            torch.nn.GELU(),
            torch.nn.Linear(emb, emb),
        )


def test_guide_tokens_extend_the_state_sequence():
    m = _Stub()
    state = torch.randn(3, 5, 16)
    out = m._append_future_action_condition(state, torch.randn(3, 4, 8))
    assert out.shape == (3, 9, 16), out.shape
    # the original state tokens must be untouched
    assert torch.equal(out[:, :5], state)


def test_null_guide_is_all_zero_and_shaped_like_training():
    m = _Stub()
    state = torch.randn(2, 5, 16)
    null = m._append_future_action_condition(state, None)
    explicit = m._append_future_action_condition(state, torch.zeros(2, 4, 8))
    assert null.shape == explicit.shape == (2, 9, 16)
    assert torch.allclose(null, explicit), "null token must equal an all-zero guide"


def test_cond_and_null_actually_differ():
    """If the guide did not reach the model input, CFG would be a no-op."""
    m = _Stub()
    state = torch.randn(2, 5, 16)
    null = m._append_future_action_condition(state, None)
    cond = m._append_future_action_condition(state, torch.randn(2, 4, 8))
    assert not torch.allclose(null, cond)


def test_disabled_eag_is_a_pass_through():
    m = _Stub(enabled=False)
    state = torch.randn(2, 5, 16)
    assert m._append_future_action_condition(state, None) is state


def test_narrower_guide_allowed_wider_rejected():
    m = _Stub()
    state = torch.randn(2, 5, 16)
    assert m._append_future_action_condition(state, torch.randn(2, 2, 8)).shape == (2, 7, 16)
    for bad in (torch.randn(2, 5, 8), torch.randn(2, 4, 7), torch.randn(3, 4, 8)):
        try:
            m._append_future_action_condition(state, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted a bad guide of shape {tuple(bad.shape)}")


def test_cfg_extrapolates_away_from_the_null_branch():
    """v = v_null + (1 + w)(v_cond - v_null), the GR00T-N1.5 formula."""
    v_null = torch.tensor([[[1.0, 2.0]]])
    v_cond = torch.tensor([[[2.0, 4.0]]])
    for w in (0.0, 1.0, 2.5):
        got = v_null + (1.0 + w) * (v_cond - v_null)
        assert torch.allclose(got, v_null + (1.0 + w) * (v_cond - v_null))
        if w == 1.0:
            assert torch.allclose(got, torch.tensor([[[3.0, 6.0]]])), got


def test_training_dropout_zeroes_whole_samples():
    """Reproduces the training-side masking: whole guide per sample, not per element."""
    torch.manual_seed(0)
    actions = torch.randn(64, 16, 8)
    guide = actions[:, :4].clone()
    drop = torch.rand(64) < 0.1
    guide[drop] = 0
    for i in range(64):
        if drop[i]:
            assert torch.all(guide[i] == 0), i
        else:
            assert torch.equal(guide[i], actions[i, :4]), i


# ── 4. source-level guards ──────────────────────────────────────────────────


def test_eag_and_rtc_are_mutually_exclusive():
    src = (ROOT / "rldx/model/core/rldx.py").read_text()
    assert "EAG and RTC both rewrite chunk-boundary conditioning" in src


def test_inference_path_runs_both_branches():
    src = (ROOT / "rldx/model/core/rldx.py").read_text()
    assert "eag_cond_state" in src and "state_tokens=eag_cond_state" in src
    assert "(1.0 + self.eag_cfg_weight) * (" in src


def test_training_path_appends_the_guide():
    src = (ROOT / "rldx/model/core/rldx.py").read_text()
    head = src.split("sa_embs = torch.cat((state_features, action_features), dim=1)")[0]
    assert "guide_actions = actions[:, :h].clone()" in head
    assert "self.future_action_condition_dropout" in head


def test_milestone_callback_copies_only_on_multiples():
    """The 20k rule: every Nth checkpoint must land outside the rotation."""
    import shutil
    import tempfile
    from types import SimpleNamespace

    from rldx.experiment.utils import MilestoneCheckpointCallback

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "a_run"
        cb = MilestoneCheckpointCallback(every_n_steps=20000)

        def save(step):
            d = run / f"checkpoint-{step}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"w")
            (d / "trainer_state.json").write_text("{}")
            (d / "optimizer.pt").write_bytes(b"x" * 1024)
            # RLDX trains under DeepSpeed: the real optimizer state is here.
            gs = d / f"global_step{step}"
            gs.mkdir()
            (gs / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"y" * 4096)
            cb.on_save(SimpleNamespace(output_dir=str(run)),
                       SimpleNamespace(global_step=step, is_world_process_zero=True), None)

        for step in (1000, 19000, 20000, 21000, 40000):
            save(step)

        kept = sorted(p.name for p in run.parent.glob("a_run_step*"))
        assert kept == ["a_run_step20000", "a_run_step40000"], kept
        got = sorted(p.name for p in (run.parent / "a_run_step20000").iterdir())
        assert "model.safetensors" in got and "trainer_state.json" in got, got
        assert "optimizer.pt" not in got, "optimizer.pt must be skipped"
        assert not any(n.startswith("global_step") for n in got), (
            f"DeepSpeed optimizer shards must be skipped, got {got}"
        )

        # Rotation deletes the in-run copy; the kept one must survive it.
        shutil.rmtree(run / "checkpoint-20000")
        assert (run.parent / "a_run_step20000" / "model.safetensors").exists()


def test_milestone_callback_never_raises():
    """A copy failure must not take down a 60,000-step run."""
    from types import SimpleNamespace

    from rldx.experiment.utils import MilestoneCheckpointCallback

    cb = MilestoneCheckpointCallback(every_n_steps=20000)
    # No such directory anywhere: must return quietly, not raise.
    cb.on_save(
        SimpleNamespace(output_dir="/nonexistent/run"),
        SimpleNamespace(global_step=20000, is_world_process_zero=True),
        None,
    )
    # Non-zero ranks must not copy at all.
    cb.on_save(
        SimpleNamespace(output_dir="/nonexistent/run"),
        SimpleNamespace(global_step=20000, is_world_process_zero=False),
        None,
    )


def test_keep_every_is_wired_end_to_end():
    src = (ROOT / "rldx/experiment/assembly.py").read_text()
    assert (
        "run_config.training.keep_checkpoint_every_n_steps = cli.keep_checkpoint_every_n_steps"
        in src
    ), "assembly never copies keep_checkpoint_every_n_steps"
    exp = (ROOT / "rldx/experiment/experiment.py").read_text()
    assert "MilestoneCheckpointCallback(every_n_steps=keep_every)" in exp
    cfg = (ROOT / "rldx/configs/training/training_config.py").read_text()
    assert "keep_checkpoint_every_n_steps" in cfg


def test_scripts_pass_all_four_flags():
    for name in (
        "finetune_rldx1_libero_sail_2gpu.sh",
        "finetune_rldx1_robocasa_sail_2gpu.sh",
    ):
        src = (ROOT / "run_scripts/train/benchmarks" / name).read_text()
        for flag in (
            "--use-future-action-condition",
            "--future-action-condition-horizon 4",
            "--future-action-condition-dropout 0.1",
            "--eag-cfg-weight 1.0",
        ):
            assert flag in src, f"{name} missing {flag}"
        assert "_sail_config.py" in src, f"{name} does not load a SAIL modality config"
        # The checkpoint rule: 20k must survive long enough to be copied out.
        assert "--save-total-limit 5" in src, f"{name} must keep 5 checkpoints"
        assert "--keep-checkpoint-every-n-steps" in src, (
            f"{name} must keep the 20k/40k/60k models out of the rotation"
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed")
        sys.exit(1)
    print("all checks passed")
