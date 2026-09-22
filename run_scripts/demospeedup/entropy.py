"""DemoSpeedup's released bandwidth-one Gaussian KDE and causal aggregation."""
from collections import deque
import numpy as np


def kde_entropy_bw1(samples):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [samples, real_action_dims]')
    distances = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    density = np.exp(-distances / 2).mean(-1)
    return float(-np.log(density + 1e-8).mean())


class CausalEntropy:
    def __init__(self, horizon):
        self.history = deque(maxlen=horizon)
        self.next_frame = 0

    def append(self, frame, samples):
        if frame != self.next_frame:
            raise ValueError('Temporal aggregation requires every frame in order')
        self.next_frame += 1
        self.history.append((frame, samples))
        pooled = np.concatenate([chunk[:, frame - start] for start, chunk in self.history])
        current = samples[:, 0]
        return dict(kde_bw1=kde_entropy_bw1(current), kde_bw1_agg=kde_entropy_bw1(pooled),
                    std_mean=float(current.std(axis=0, ddof=1).mean()),
                    std_mean_agg=float(pooled.std(axis=0, ddof=1).mean()),
                    aggregate_samples=len(pooled))
