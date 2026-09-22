"""Check KDE against the released formula and temporal chunk alignment."""
import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run_scripts/demospeedup'))
from entropy import CausalEntropy, kde_entropy_bw1
from plan_resume import validate_episode


class EntropyTests(unittest.TestCase):
    def test_resume_rejects_incomplete_or_nonfinite_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.parquet'
            data = pd.DataFrame(dict(episode=[3, 3], frame=[0, 1], kde_bw1=[1., 2.],
                kde_bw1_agg=[1., 2.], std_mean=[.1, .2], std_mean_agg=[.1, .2], entropy_z=[-1., 1.]))
            data.to_parquet(path)
            validate_episode(path, 3, 2)
            with self.assertRaises(ValueError):
                validate_episode(path, 3, 3)
            with self.assertRaises(ValueError):
                validate_episode(path, 4, 2)
            data.loc[0, 'kde_bw1_agg'] = np.nan
            data.to_parquet(path)
            with self.assertRaises(ValueError):
                validate_episode(path, 3, 2)

    def test_released_kde_formula(self):
        x = np.random.default_rng(7).normal(size=(160, 7)).astype(np.float32)
        t = torch.from_numpy(x)[None]
        kernel = torch.exp(-((t.unsqueeze(2) - t.unsqueeze(1)) ** 2).sum(-1) / 2)
        expected = -torch.log(kernel.sum(2) / 160 + 1e-8).mean().item()
        self.assertAlmostEqual(kde_entropy_bw1(x), expected, places=5)

    def test_causal_offsets_and_eviction(self):
        chunks = np.random.default_rng(2).normal(size=(5, 10, 3, 7)).astype(np.float32)
        agg = CausalEntropy(3)
        for frame, chunk in enumerate(chunks):
            got = agg.append(frame, chunk)
            pool = np.concatenate([chunks[start, :, frame - start] for start in range(max(0, frame-2), frame+1)])
            self.assertEqual(got['aggregate_samples'], min(frame+1, 3)*10)
            self.assertAlmostEqual(got['kde_bw1_agg'], kde_entropy_bw1(pool), places=6)
        with self.assertRaises(ValueError):
            agg.append(6, chunks[0])
        self.assertEqual(CausalEntropy(3).append(0, chunks[0])['aggregate_samples'], 10)


if __name__ == '__main__':
    unittest.main()
