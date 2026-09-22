# SPDX-License-Identifier: Apache-2.0
"""ATQ label-gated variable-horizon MoE — pure-torch helpers.

Ported from GR00T-action-quantization (branch ``jimin-dev-ATQ-finalized-only``,
``gr00t/model/action_head/flow_matching_action_head.py`` and
``rotation_gt.py``) onto RLDX-1's MSAT action model.  Everything in this file
is framework-free (torch + math only) so it can be unit-tested on CPU without
the backbone, and so the model file only carries the wiring.

Expert grid (index order is load-bearing: the group masks below index into it)::

    idx  name  source steps  rows at 2.0x  rows at 2.5x  group
      0  main      H=16         16            16          fine
      1  m8        16            8             6 (15 covered)  compressed
      2  m4         8            4             3           compressed
      3  n8         8            8             8           fine

Compression rule: continuous dims are SUMMED over each block (delta
accumulation), ``discrete_dims`` take the block's LAST value, and a block is
valid only if every source step in it is valid.  With ``reduction="last"``
every dim takes the last value (absolute-action spaces).

Normalisation caveat (why the processor needs the block sizes at decode):
RLDX min-max maps ``x' = a*x + b``; the sum of ``k`` normalised steps is
``a*sum(x) + k*b``.  The inverse therefore needs ``k`` per row, and must not
clip to [-1, 1].  See ``rldx.data.utils.unnormalize_values_minmax_blocks``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


EXPERT_NAMES: tuple[str, ...] = ("main", "m8", "m4", "n8")
NUM_EXPERTS = len(EXPERT_NAMES)
# Expert order is [fine16, comp_from16, comp_from8, fine8]; the two groups the
# conf label chooses between are these index sets.
FINE_EXPERTS: tuple[int, ...] = (0, 3)
COMP_EXPERTS: tuple[int, ...] = (1, 2)

# Hardcoded block plans for the non-integer speeds (H=16 only).  The 2.0x grid
# is derived from H (``[2]*(H//2)``, ``[2]*(H//4)``) and needs no table entry.
#   2.5x: 16 -> [2,3,2,3,2,3] (6 rows, 15 covered; the lone tail step is dropped)
#          8 -> [2,3,3]        (3 rows, all 8 covered)
#   1.67x (LIBERO's 5/3 rhythm): 16 -> [2,2,1]*3 (9 rows, 15 covered)
#                                 9 -> [2,2,1,2,2] (5 rows)
#   3.0x: 16 -> [3]*5 (5 rows, 15 covered);  9 -> [3,3,3] (3 rows)
SPEED_BLOCK_PLANS: dict[float, tuple[list[int], list[int]]] = {
    2.5: ([2, 3, 2, 3, 2, 3], [2, 3, 3]),
    1.67: ([2, 2, 1, 2, 2, 1, 2, 2, 1], [2, 2, 1, 2, 2]),
    3.0: ([3, 3, 3, 3, 3], [3, 3, 3]),
}


# ---------------------------------------------------------------------------
# Block plans
# ---------------------------------------------------------------------------


def parse_block_plan(spec: str | None, span: int, label: str) -> list[int] | None:
    """Parse a comma-separated block plan.  Empty -> ``None`` (use the speed table).

    A plan whose sum exceeds ``span`` is rejected: it would put steps that are
    not in the batch into the target and silently train a partial last block.
    """
    spec = spec or ""
    if not isinstance(spec, str):
        raise ValueError(f"{label}: must be a string (got {type(spec).__name__})")
    spec = spec.strip()
    if not spec:
        return None
    try:
        plan = [int(x) for x in spec.replace(" ", "").split(",") if x]
    except ValueError as e:
        raise ValueError(f"{label}: must be comma-separated integers (got {spec!r})") from e
    if not plan or any(b < 1 for b in plan):
        raise ValueError(f"{label}: every block must be >= 1 (got {plan})")
    if sum(plan) > span:
        raise ValueError(f"{label}: sum {sum(plan)} exceeds span {span} ({plan})")
    return plan


def resolve_block_plans(
    action_horizon: int,
    speed: float,
    plan_full: str | None = "",
    plan_half: str | None = "",
) -> tuple[list[int], list[int]]:
    """Return ``(full_plan, half_plan)`` for the two compressed experts.

    Explicit plans win over the speed table.  Both must be given together —
    mixing one explicit plan with one table plan is almost always a mistake.
    """
    H = int(action_horizon)
    pf = parse_block_plan(plan_full, H, "atq_block_plan_full")
    ph = parse_block_plan(plan_half, H, "atq_block_plan_half")
    if (pf is None) != (ph is None):
        raise ValueError(
            "atq_block_plan_full and atq_block_plan_half must be set together "
            f"(got full={plan_full!r}, half={plan_half!r})"
        )
    if pf is not None and ph is not None:
        return pf, ph

    speed = float(speed)
    if speed == 2.0:
        if H % 4 != 0:
            raise ValueError(f"atq_speed=2.0 needs action_horizon divisible by 4 (got {H})")
        return [2] * (H // 2), [2] * (H // 4)
    if speed in SPEED_BLOCK_PLANS:
        if H != 16:
            raise ValueError(
                f"atq_speed={speed} has a hardcoded plan for action_horizon=16 only "
                f"(got {H}); pass --atq-block-plan-full/--atq-block-plan-half explicitly."
            )
        full, half = SPEED_BLOCK_PLANS[speed]
        return list(full), list(half)
    raise ValueError(
        f"Unsupported atq_speed={speed}; use one of 2.0, {sorted(SPEED_BLOCK_PLANS)} "
        "or pass explicit block plans."
    )


def expert_block_sizes(action_horizon: int, full: Sequence[int], half: Sequence[int]) -> list[list[int]]:
    """Per-expert block sizes, one entry per output row. ``[1]`` rows are uncompressed."""
    H = int(action_horizon)
    return [[1] * H, list(full), list(half), [1] * (H // 2)]


def expert_horizons(action_horizon: int, full: Sequence[int], half: Sequence[int]) -> list[int]:
    return [len(s) for s in expert_block_sizes(action_horizon, full, half)]


# ---------------------------------------------------------------------------
# Target compression
# ---------------------------------------------------------------------------


def compress_actions_blocks(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    sizes: Sequence[int],
    discrete_dims: Iterable[int] = (),
    reduction: str = "sum",
    rotation_gt: "RotationGT | None" = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress ``actions[:, :sum(sizes)]`` block-wise into ``len(sizes)`` rows.

    Args:
        actions: ``(B, T, D)`` normalised clean actions (T >= sum(sizes)).
        action_mask: ``(B, T, D)`` validity mask (bool or {0,1} float).
        sizes: block sizes, e.g. ``[2]*8`` or ``[2,3,2,3,2,3]``.
        discrete_dims: action dims that take the block's LAST value under
            ``reduction="sum"`` (gripper / control-mode style signals).
        reduction: ``"sum"`` (delta spaces) or ``"last"`` (absolute spaces).
        rotation_gt: optional :class:`RotationGT` that replaces the summed
            rotation dims with the SO(3) composition of the block.

    Returns:
        ``(rows, rows_mask)`` with shapes ``(B, len(sizes), D)``; ``rows_mask``
        has the dtype of ``action_mask``.
    """
    if reduction not in ("sum", "last"):
        raise ValueError(f"reduction must be 'sum' or 'last', got {reduction!r}")
    B, T, D = actions.shape
    total = int(sum(sizes))
    if total > T:
        raise ValueError(f"block sizes {list(sizes)} need {total} steps, have {T}")
    mask_bool = action_mask.bool()
    disc = [int(d) for d in discrete_dims if int(d) < D]
    rows, masks, pos = [], [], 0
    for n in sizes:
        grp = actions[:, pos : pos + n]
        grp_mask = mask_bool[:, pos : pos + n]
        if reduction == "last":
            row = grp[:, -1, :]
        else:
            row = grp.sum(dim=1)
            if disc:
                row = row.clone()
                row[..., disc] = grp[:, -1, :][..., disc]
        merged_mask = grp_mask.all(dim=1)
        if rotation_gt is not None:
            if reduction != "sum":
                raise ValueError("SO(3) delta targets cannot be used with reduction='last'")
            merged, merged_mask = rotation_gt.apply(grp[:, None], row[:, None].clone(), merged_mask[:, None].clone())
            row, merged_mask = merged[:, 0], merged_mask[:, 0]
        rows.append(row)
        masks.append(merged_mask)
        pos += n
    return torch.stack(rows, dim=1), torch.stack(masks, dim=1).to(action_mask.dtype)


def expert_targets(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    action_horizon: int,
    full: Sequence[int],
    half: Sequence[int],
    discrete_dims: Iterable[int] = (),
    reduction: str = "sum",
    rotation_gt: "RotationGT | None" = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-expert ``(clean, mask)`` supervision in ``EXPERT_NAMES`` order.

    0: main  -> ``actions[:, :H]``
    1: m8    -> blocks ``full`` over ``actions[:, :sum(full)]``
    2: m4    -> blocks ``half`` over ``actions[:, :sum(half)]``
    3: n8    -> ``actions[:, :H//2]``
    """
    H = int(action_horizon)
    H8 = H // 2
    c1, m1 = actions[:, :H], action_mask[:, :H]
    c2, m2 = compress_actions_blocks(
        actions[:, : sum(full)], action_mask[:, : sum(full)], full, discrete_dims, reduction, rotation_gt
    )
    c3, m3 = compress_actions_blocks(
        actions[:, : sum(half)], action_mask[:, : sum(half)], half, discrete_dims, reduction, rotation_gt
    )
    c4, m4 = actions[:, :H8], action_mask[:, :H8]
    return [(c1, m1), (c2, m2), (c3, m3), (c4, m4)]


# ---------------------------------------------------------------------------
# Label gate
# ---------------------------------------------------------------------------


def group_mask(compress: torch.Tensor, n_experts: int, device, dtype) -> torch.Tensor:
    """``1.0`` on the experts of the group the conf decision picked, ``0.0`` elsewhere.

    ``compress`` is a ``(B,)`` bool tensor: True where the chunk is compressible.
    """
    mask = torch.zeros(compress.shape[0], n_experts, device=device, dtype=dtype)
    fine = [i for i in FINE_EXPERTS if i < n_experts]
    comp = [i for i in COMP_EXPERTS if i < n_experts]
    keep_fine = (~compress).to(dtype).unsqueeze(-1)
    keep_comp = compress.to(dtype).unsqueeze(-1)
    mask[:, fine] = keep_fine.expand(-1, len(fine))
    mask[:, comp] = keep_comp.expand(-1, len(comp))
    return mask


def apply_group_mask(router_probs: torch.Tensor, compress: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero the out-of-group experts and renormalise what is left.

    Softmax ratios ``p_i/p_j = exp(z_i - z_j)`` do not depend on the other
    logits, so the within-group preference learned with 4-way training is
    preserved by masking + renormalising.
    """
    mask = group_mask(compress, router_probs.shape[-1], router_probs.device, router_probs.dtype)
    gated = router_probs * mask
    return gated / gated.sum(dim=-1, keepdim=True).clamp(min=eps)


# ---------------------------------------------------------------------------
# SO(3) merged rotation targets
# ---------------------------------------------------------------------------


class RotationGT:
    """SO(3) merged targets in *normalised* action space (OSC spatial increments).

    ``spec`` keys:
        indices            three action dims holding the axis-angle delta
        raw_scale/offset   normalised -> raw:  raw = z * raw_scale + raw_offset
        controller_scale/offset  raw -> radians: r = raw * cs + cb
        frame              must be ``"spatial"`` (fixed reference frame)
        action_key         informational

    The composed rotation is mapped back into the *1x* normalised
    parametrisation, so these dims decode with ``k=1`` (no block-sum bias).
    No gradient graph, no host copies.
    """

    def __init__(self, spec: dict):
        self.spec = dict(spec)
        self.indices = [int(i) for i in spec["indices"]]
        if len(self.indices) != 3 or len(set(self.indices)) != 3 or min(self.indices) < 0:
            raise ValueError("SO3 requires three distinct rotation dimensions")
        for name in ("raw_scale", "raw_offset", "controller_scale", "controller_offset"):
            if len(spec[name]) != 3 or not all(math.isfinite(float(x)) for x in spec[name]):
                raise ValueError(f"Invalid {name}")
        if any(float(x) <= 0 for x in spec["raw_scale"]):
            raise ValueError("Constant rotation dimensions cannot represent SO3 cross-axis corrections")
        if any(float(x) == 0 for x in spec["controller_scale"]):
            raise ValueError("Controller rotation scale must be nonzero")
        if spec.get("frame") != "spatial":
            raise ValueError("Only fixed-reference-frame OSC spatial increments are supported")
        self._cache: dict = {}

    @staticmethod
    def _multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Hamilton product, scalar-first; p*q applies q then p.
        pw, pv = p[..., :1], p[..., 1:]
        qw, qv = q[..., :1], q[..., 1:]
        return torch.cat(
            (pw * qw - (pv * qv).sum(-1, keepdim=True), pw * qv + qw * pv + torch.linalg.cross(pv, qv, dim=-1)),
            dim=-1,
        )

    @torch.no_grad()
    def apply(self, grouped: torch.Tensor, compressed: torch.Tensor, mask: torch.Tensor):
        """Replace ``compressed[..., indices]`` with the SO(3) composition of ``grouped``.

        Args:
            grouped: ``(..., k, D)`` normalised source steps of one block.
            compressed: ``(..., D)`` block row (already summed); modified in place.
            mask: ``(..., D)`` bool validity; rotation dims are coupled, so a
                missing axis invalidates all three merged axes.
        """
        dtype = torch.float64 if grouped.dtype == torch.float64 else torch.float32
        cache_key = (grouped.device, dtype)
        if cache_key not in self._cache:
            self._cache[cache_key] = tuple(
                torch.tensor(self.spec[n], device=grouped.device, dtype=dtype)
                for n in ("raw_scale", "raw_offset", "controller_scale", "controller_offset")
            )
        ns, nb, cs, cb = self._cache[cache_key]
        if max(self.indices) >= grouped.shape[-1]:
            raise ValueError("Rotation indices exceed action dimension")
        with torch.autocast(device_type=grouped.device.type, enabled=False):
            z = grouped[..., self.indices].to(dtype)
            r = (z * ns + nb) * cs + cb
            angle = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
            q = torch.cat((torch.cos(angle / 2), r * (0.5 * torch.sinc(angle / (2 * math.pi)))), dim=-1)
            total = q[..., 0, :]
            for i in range(1, grouped.shape[-2]):
                total = self._multiply(q[..., i, :], total)
            total = total / torch.linalg.vector_norm(total, dim=-1, keepdim=True).clamp_min(1e-15)
            total = torch.where(total[..., :1] < 0, -total, total)
            v = total[..., 1:]
            vnorm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            angle = 2 * torch.atan2(vnorm, total[..., :1])
            # angle / sin(angle/2), with its continuous limit at zero.
            rot = v * (2 / torch.sinc(angle / (2 * math.pi)))
            raw = (rot - cb) / cs
            normalized = (raw - nb) / ns
        compressed[..., self.indices] = normalized.to(compressed.dtype)
        valid = mask[..., self.indices].all(dim=-1, keepdim=True)
        mask[..., self.indices] = valid.expand_as(mask[..., self.indices])
        return compressed, mask


def build_rotation_spec(
    min_vals: Sequence[float],
    max_vals: Sequence[float],
    indices: Sequence[int],
    controller_scale: float,
    action_key: str,
) -> dict:
    """Build a :class:`RotationGT` spec from the normaliser bounds of the rotation key.

    ``min_vals``/``max_vals`` are the bounds the min-max normaliser actually
    uses (q01/q99 when ``use_percentiles`` is on).  Under RLDX's
    ``x' = 2(x-min)/(max-min) - 1`` these give ``raw = z*(max-min)/2 + (max+min)/2``.
    """
    lo = [float(x) for x in min_vals]
    hi = [float(x) for x in max_vals]
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("Expected three axis-angle action dimensions")
    return dict(
        version=1,
        indices=[int(i) for i in indices],
        raw_scale=[(h - l) / 2 for l, h in zip(lo, hi)],
        raw_offset=[(h + l) / 2 for l, h in zip(lo, hi)],
        controller_scale=[float(controller_scale)] * 3,
        controller_offset=[0.0] * 3,
        frame="spatial",
        action_key=str(action_key),
    )


def rotation_gt_from_spec(spec) -> RotationGT | None:
    return RotationGT(spec) if isinstance(spec, dict) and spec else None
