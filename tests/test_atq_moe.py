"""CPU unit tests for the ATQ label-gated MoE helpers.

Run inside the training venv (needs torch + numpy):

    .venv/bin/python -m unittest tests/test_atq_moe.py -v

Covers the pure-torch target construction (block plans, sum/last/discrete
compression, SO(3) rotation composition), the label group mask, and the
k-corrected unnormalisation the processor uses to decode compressed rows.
"""

import importlib.util
import math
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None

ROOT = Path(__file__).parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch/numpy not installed")
class BlockPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atq = _load("atq", "rldx/model/modules/action_model/atq.py")

    def test_speed_2_derives_from_horizon(self):
        full, half = self.atq.resolve_block_plans(16, 2.0)
        self.assertEqual(full, [2] * 8)
        self.assertEqual(half, [2] * 4)
        self.assertEqual(self.atq.expert_horizons(16, full, half), [16, 8, 4, 8])

    def test_speed_2_5_table(self):
        full, half = self.atq.resolve_block_plans(16, 2.5)
        self.assertEqual(full, [2, 3, 2, 3, 2, 3])
        self.assertEqual(half, [2, 3, 3])
        self.assertEqual(self.atq.expert_horizons(16, full, half), [16, 6, 3, 8])

    def test_explicit_plans_win(self):
        full, half = self.atq.resolve_block_plans(16, 2.5, "2,2,1,2,2,1,2,2,2", "2,2,1,2,1")
        self.assertEqual(full, [2, 2, 1, 2, 2, 1, 2, 2, 2])
        self.assertEqual(half, [2, 2, 1, 2, 1])

    def test_rejects_bad_plans(self):
        with self.assertRaises(ValueError):
            self.atq.parse_block_plan("2,2,2,2,2,2,2,2,2", 16, "x")  # sum 18 > 16
        with self.assertRaises(ValueError):
            self.atq.parse_block_plan("2,0,2", 16, "x")
        with self.assertRaises(ValueError):
            self.atq.resolve_block_plans(16, 2.5, "2,3", "")  # one plan only
        with self.assertRaises(ValueError):
            self.atq.resolve_block_plans(12, 2.5)  # table is H=16 only
        with self.assertRaises(ValueError):
            self.atq.resolve_block_plans(16, 1.5)


@unittest.skipIf(torch is None, "torch/numpy not installed")
class CompressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atq = _load("atq", "rldx/model/modules/action_model/atq.py")

    def _data(self, B=2, T=16, D=8):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(B, T, D, generator=g, dtype=torch.float64)
        m = torch.ones(B, T, D, dtype=torch.float32)
        return x, m

    def test_sum_with_discrete_last(self):
        x, m = self._data()
        rows, mask = self.atq.compress_actions_blocks(x, m, [2] * 8, discrete_dims=[6])
        expected = x.reshape(2, 8, 2, 8).sum(2)
        expected[..., 6] = x[:, 1:16:2, 6]
        torch.testing.assert_close(rows, expected)
        self.assertEqual(mask.dtype, m.dtype)
        self.assertTrue(bool(mask.all()))

    def test_last_reduction(self):
        x, m = self._data()
        rows, _ = self.atq.compress_actions_blocks(x, m, [2] * 8, reduction="last")
        torch.testing.assert_close(rows, x[:, 1:16:2])

    def test_block_plan_2_5(self):
        x, m = self._data()
        rows, _ = self.atq.compress_actions_blocks(x, m, [2, 3, 2, 3, 2, 3], discrete_dims=[6])
        self.assertEqual(rows.shape, (2, 6, 8))
        torch.testing.assert_close(rows[:, 1, :6], x[:, 2:5, :6].sum(1))
        torch.testing.assert_close(rows[:, 1, 6], x[:, 4, 6])
        expected_last = x[:, 12:15].sum(1)
        expected_last[:, 6] = x[:, 14, 6]
        torch.testing.assert_close(rows[:, 5], expected_last)

    def test_partial_block_invalidates_row(self):
        x, m = self._data()
        m[0, 3, 2] = 0.0
        _, mask = self.atq.compress_actions_blocks(x, m, [2] * 8)
        self.assertEqual(mask[0, 1, 2].item(), 0.0)
        self.assertEqual(mask[0, 1, 3].item(), 1.0)
        self.assertEqual(mask[1, 1, 2].item(), 1.0)

    def test_expert_targets_shapes(self):
        x, m = self._data()
        full, half = self.atq.resolve_block_plans(16, 2.5)
        targets = self.atq.expert_targets(x, m, 16, full, half, discrete_dims=[6])
        self.assertEqual([t[0].shape[1] for t in targets], [16, 6, 3, 8])
        torch.testing.assert_close(targets[0][0], x)
        torch.testing.assert_close(targets[3][0], x[:, :8])

    def test_padded_dims_stay_masked(self):
        x, m = self._data()
        m[..., 5:] = 0.0  # simulate max_action_dim padding
        _, mask = self.atq.compress_actions_blocks(x, m, [2] * 8)
        self.assertTrue(bool((mask[..., 5:] == 0).all()))
        self.assertTrue(bool((mask[..., :5] == 1).all()))


@unittest.skipIf(torch is None, "torch/numpy not installed")
class GroupMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atq = _load("atq", "rldx/model/modules/action_model/atq.py")

    def test_mask_preserves_within_group_ratio(self):
        logits = torch.tensor([[2.0, 0.5, 1.5, 0.1], [0.3, 2.0, 0.2, 1.0]])
        probs = torch.softmax(logits, -1)
        compress = torch.tensor([True, False])
        gated = self.atq.apply_group_mask(probs, compress)
        torch.testing.assert_close(gated.sum(-1), torch.ones(2))
        # sample 0 compressed -> only m8 (1), m4 (2) survive with ratio exp(0.5-1.5)
        self.assertEqual(gated[0, 0].item(), 0.0)
        self.assertEqual(gated[0, 3].item(), 0.0)
        self.assertAlmostEqual((gated[0, 1] / gated[0, 2]).item(), math.exp(0.5 - 1.5), places=5)
        # sample 1 fine -> only main (0), n8 (3)
        self.assertEqual(gated[1, 1].item(), 0.0)
        self.assertEqual(gated[1, 2].item(), 0.0)
        self.assertAlmostEqual((gated[1, 0] / gated[1, 3]).item(), math.exp(0.3 - 1.0), places=5)


@unittest.skipIf(torch is None, "torch/numpy not installed")
class RotationGTTest(unittest.TestCase):
    """Independent Rodrigues-matrix oracle for the SO(3) composition."""

    @classmethod
    def setUpClass(cls):
        cls.atq = _load("atq", "rldx/model/modules/action_model/atq.py")
        cls.spec = dict(
            indices=[3, 4, 5],
            raw_scale=[0.7, 0.9, 1.1],
            raw_offset=[0.11, -0.23, 0.07],
            controller_scale=[0.5] * 3,
            controller_offset=[0.0] * 3,
            frame="spatial",
            action_key="end_effector_rotation",
        )

    @staticmethod
    def _rotvec_to_mat(r):
        theta = np.linalg.norm(r)
        if theta < 1e-12:
            return np.eye(3)
        k = r / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K

    @staticmethod
    def _mat_to_rotvec(R):
        cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        theta = math.acos(cos)
        if theta < 1e-12:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(theta))
        return axis * theta

    def test_matches_sequential_composition(self):
        rng = np.random.default_rng(7)
        for k in (1, 2, 3, 4):
            a = rng.normal(0, 0.2, (3, k, 9))
            x = torch.tensor(a, dtype=torch.float64)
            summed = x.sum(1)
            mask = torch.ones((3, 9), dtype=torch.bool)
            rows, m = self.atq.RotationGT(self.spec).apply(x, summed.clone(), mask.clone())
            raw = a[..., 3:6] * np.array(self.spec["raw_scale"]) + np.array(self.spec["raw_offset"])
            for b in range(3):
                R = np.eye(3)
                for i in range(k):
                    R = self._rotvec_to_mat(raw[b, i] * 0.5) @ R  # later step applied after
                expected = (self._mat_to_rotvec(R) / 0.5 - np.array(self.spec["raw_offset"])) / np.array(
                    self.spec["raw_scale"]
                )
                np.testing.assert_allclose(rows[b, 3:6].numpy(), expected, atol=1e-9)
            torch.testing.assert_close(rows[:, :3], summed[:, :3])
            torch.testing.assert_close(rows[:, 6:], summed[:, 6:])
            self.assertTrue(bool(m.all()))

    def test_coupled_mask_and_zero(self):
        spec = dict(self.spec, raw_scale=[1.0] * 3, raw_offset=[0.0] * 3, controller_scale=[1.0] * 3)
        x = torch.zeros(2, 2, 7, dtype=torch.float64)
        x[0, 0, 3] = math.pi / 2
        x[0, 1, 4] = math.pi / 2
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, 4] = False
        rows, m = self.atq.RotationGT(spec).apply(x, x.sum(1), mask)
        self.assertFalse(bool(m[0, 3:6].any()))
        self.assertTrue(bool(m[0, :3].all()))
        self.assertTrue(bool(torch.isfinite(rows).all()))
        zero = torch.zeros_like(x)
        out, _ = self.atq.RotationGT(spec).apply(zero, zero.sum(1), torch.ones(2, 7, dtype=torch.bool))
        self.assertEqual(int(torch.count_nonzero(out)), 0)

    def test_through_block_compression(self):
        atq = self.atq
        gt = atq.RotationGT(self.spec)
        x = torch.randn(2, 16, 8, dtype=torch.float64)
        m = torch.ones(2, 16, 8)
        rows, mask = atq.compress_actions_blocks(x, m, [2] * 8, discrete_dims=[6], rotation_gt=gt)
        plain, _ = atq.compress_actions_blocks(x, m, [2] * 8, discrete_dims=[6])
        torch.testing.assert_close(rows[..., :3], plain[..., :3])
        torch.testing.assert_close(rows[..., 6:], plain[..., 6:])
        self.assertFalse(torch.allclose(rows[..., 3:6], plain[..., 3:6]))
        with self.assertRaises(ValueError):
            atq.compress_actions_blocks(x, m, [2] * 8, reduction="last", rotation_gt=gt)

    def test_invalid_specs(self):
        with self.assertRaises(ValueError):
            self.atq.RotationGT(dict(self.spec, raw_scale=[0.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            self.atq.RotationGT(dict(self.spec, frame="body"))

    def test_build_spec_from_bounds(self):
        spec = self.atq.build_rotation_spec([-0.2, -0.4, -0.6], [0.3, 0.5, 0.7], [3, 4, 5], 0.5, "rot")
        np.testing.assert_allclose(spec["raw_scale"], [0.25, 0.45, 0.65])
        np.testing.assert_allclose(spec["raw_offset"], [0.05, 0.05, 0.05])
        self.assertEqual(spec["indices"], [3, 4, 5])


@unittest.skipIf(torch is None, "torch/numpy not installed")
class BlockUnnormalizeTest(unittest.TestCase):
    """The k-corrected inverse must undo 'normalise each step, then sum k steps'."""

    @classmethod
    def setUpClass(cls):
        cls.utils = _load("rldx_data_utils", "rldx/data/utils.py")

    def test_minmax_roundtrip(self):
        rng = np.random.default_rng(3)
        lo = np.array([-0.4, -0.2, 0.0, 1.0])
        hi = np.array([0.6, 0.2, 1.0, 1.0])  # last dim constant
        params = {"min": lo, "max": hi}
        raw = rng.uniform(lo, hi, size=(16, 4))
        norm = self.utils.normalize_values_minmax(raw, params)
        sizes = [2, 3, 2, 3, 2, 3]
        rows, pos = [], 0
        for k in sizes:
            row = norm[pos : pos + k].sum(0)
            row[2] = norm[pos + k - 1, 2]  # discrete dim: last of block
            rows.append(row)
            pos += k
        rows = np.stack(rows)
        out = self.utils.unnormalize_values_minmax_blocks(rows, params, sizes, exempt_dims=[2])
        pos = 0
        for t, k in enumerate(sizes):
            np.testing.assert_allclose(out[t, :2], raw[pos : pos + k, :2].sum(0), atol=1e-9)
            np.testing.assert_allclose(out[t, 2], raw[pos + k - 1, 2], atol=1e-9)
            np.testing.assert_allclose(out[t, 3], k * 1.0, atol=1e-9)  # constant dim sums to k*const
            pos += k

    def test_minmax_k1_matches_plain_inverse_without_clip(self):
        params = {"min": np.array([-1.0, 0.0]), "max": np.array([1.0, 2.0])}
        y = np.array([[0.5, -0.5], [1.5, 0.0]])  # second row outside [-1, 1]
        out = self.utils.unnormalize_values_minmax_blocks(y, params, [1, 1])
        np.testing.assert_allclose(out[0], self.utils.unnormalize_values_minmax(y[:1], params)[0])
        np.testing.assert_allclose(out[1], [1.5, 1.0])  # unclipped

    def test_meanstd_roundtrip(self):
        rng = np.random.default_rng(5)
        params = {"mean": np.array([0.3, -1.0, 2.0]), "std": np.array([0.5, 2.0, 0.0])}
        raw = rng.normal(size=(8, 3))
        norm = self.utils.normalize_values_meanstd(raw, params)
        sizes = [2, 2, 4]
        rows = np.stack([norm[0:2].sum(0), norm[2:4].sum(0), norm[4:8].sum(0)])
        out = self.utils.unnormalize_values_meanstd_blocks(rows, params, sizes)
        np.testing.assert_allclose(out[0], raw[0:2].sum(0), atol=1e-9)
        np.testing.assert_allclose(out[2], raw[4:8].sum(0), atol=1e-9)

    def test_batched_input(self):
        params = {"min": np.array([-1.0]), "max": np.array([3.0])}
        y = np.zeros((2, 3, 1))
        out = self.utils.unnormalize_values_minmax_blocks(y, params, [2, 2, 2])
        np.testing.assert_allclose(out, np.full((2, 3, 1), 2 * 1.0))  # k * (max+min)/2


if __name__ == "__main__":
    unittest.main()
