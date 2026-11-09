#!/usr/bin/env python3
"""Synthesize the claw-rubbing gesture (a villain's snicker) from hand-placed keyframes.

    python tools/synth_rub_gesture.py

Writes `tasks/jumper/gesture_rub/media/rub.npz` in the layout
`tools/import_wbc_gestures.py` writes, so `jumper.gesture_rub` trains on the dance
machinery exactly as the four imported gestures do.

## The motion

The robot raises both claws in front of its chest, pressed together, and rubs them:
the two arms swing in antiphase about the shoulder pitch, so one claw slides up
while the other slides down. The jaws chatter open and shut, and the body shakes
three times ("heh heh heh") with all four legs bobbing in phase.

The pose the rubbing happens around is not invented: it is the row of
`tasks/jumper/gesture_bow/media/bow.npz` where bow holds its claws together in front
of its chest with the body level (t = 2.8 s), with the two shoulder yaws opened by
`GAP` so that the claws touch lightly rather than press -- bow's docstring notes
its claws press into each other there.

The lead-in is bow's own rise from HOME to that row and the lead-out is the same
rise reversed; the hold at HOME is `import_wbc_gestures`'s, and so is the base pose
(`solve_resting_pose`): six feet, rigid, least sliding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import import_wbc_dances as dances
import import_wbc_gestures as gestures
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DT = 0.02

#: bow.npz row with the claws together and the body level.
BOW_ROW = 1400

#: Each shoulder yaw opened by this much from bow's, rad, so the claws touch, not press.
GAP = 0.18

#: Rubbing: antiphase shoulder-pitch swing, rad and Hz, for RUB_S seconds.
RUB_AMP = 0.14
RUB_HZ = 2.0
RUB_S = 3.0
#: Ramp the rubbing amplitude in and out over this long, seconds.
RUB_RAMP_S = 0.3

#: Jaw chatter: amplitude rad (towards open), Hz.
JAW_AMP = 0.18
JAW_HZ = 4.0

#: The snicker: knee bob of the four legs, rad and Hz, in three bursts.
SHAKE_AMP = 0.06
SHAKE_HZ = 6.0
SHAKE_BURSTS = ((0.6, 1.1), (1.4, 1.9), (2.2, 2.8))


def _window(t: np.ndarray, a: float, b: float, ramp: float) -> np.ndarray:
    """1 inside [a, b], smoothstepped to 0 over `ramp` at each end."""
    up = np.clip((t - a) / ramp, 0, 1)
    down = np.clip((b - t) / ramp, 0, 1)
    return gestures._smoothstep(np.minimum(up, down))


def build() -> np.ndarray:
    keys = dances._keys()
    col = {k: i for i, k in enumerate(keys)}
    bow = np.load(REPO / "tasks/jumper/gesture_bow/media/bow.npz")
    base = np.array([bow[k][BOW_ROW] for k in keys], dtype=np.float64)
    # L0 is the left arm, L5 the right; J0 yaw is mirrored (left negative).

    t = np.arange(round(RUB_S / DT)) * DT
    q = np.repeat(base[None], len(t), axis=0)
    env = _window(t, 0.0, RUB_S, RUB_RAMP_S)
    q[:, col["L0_j0"]] += GAP
    q[:, col["L5_j0"]] -= GAP
    rub = RUB_AMP * env * np.sin(2 * np.pi * RUB_HZ * t)
    # Shoulder pitch is not mirrored between the arms (both negative raised), so
    # opposite signs are antiphase.
    q[:, col["L0_j1"]] += rub
    q[:, col["L5_j1"]] -= rub
    jaw = JAW_AMP * env * 0.5 * (1 - np.cos(2 * np.pi * JAW_HZ * t))
    # Left jaw range [-1.6, 0.1], right [-0.1, 1.6]: opening is away from 0.1 / -0.1.
    q[:, col["L0_j4"]] -= jaw
    q[:, col["L5_j4"]] += jaw
    shake = sum(_window(t, a, b, 0.1) for a, b in SHAKE_BURSTS)
    bob = SHAKE_AMP * shake * np.sin(2 * np.pi * SHAKE_HZ * t)
    # Knees: L1 LM, L2 LR (positive side), L3 RR, L4 RM (mirrored).
    for leg, sign in (("L1", 1), ("L2", 1), ("L3", -1), ("L4", -1)):
        q[:, col[f"{leg}_j1"]] += sign * bob
    return q


def bow_rise() -> np.ndarray:
    """bow.npz from HOME up to `BOW_ROW`: bow's own way of getting its claws up."""
    bow = np.load(REPO / "tasks/jumper/gesture_bow/media/bow.npz")
    # bow is stored at its own rate; take every `step`-th row to land on `DT`.
    step = round(DT / float(bow["dt"]))
    rows = np.arange(BOW_ROW % step, BOW_ROW, step)
    rise = np.stack([bow[k][rows] for k in dances._keys()], axis=1).astype(np.float64)
    # Open the claws by GAP on the way up, so they arrive touching, not pressing.
    keys = dances._keys()
    u = gestures._smoothstep(np.linspace(0.0, 1.0, len(rise)))
    rise[:, keys.index("L0_j0")] += GAP * u
    rise[:, keys.index("L5_j0")] -= GAP * u
    return rise


def main() -> int:
    from tasks.jumper.common.constants import HOME

    robot = gestures._Robot()
    q = build()
    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - q.min(axis=0), q.max(axis=0) - hi)
    for i in np.nonzero(over > 0)[0]:
        print(f"  clamped {robot.names[i]} by up to {over[i] * 1000:.1f} mrad")
    q = np.clip(q, lo, hi)

    home = np.array([HOME[n] for n in robot.names])
    # A straight lead from HOME raises the claws while they are still on the floor,
    # and the rigid solver tips the robot 50 degrees onto them. bow's own rise does
    # not, so the clip rises as bow does and sits back down as bow rose, reversed.
    rise = bow_rise()
    if np.abs(rise[0] - home).max() > 1e-3 or np.abs(rise[-1] - q[0]).max() > 0.02:
        raise ValueError("bow.npz no longer starts at HOME or reaches BOW_ROW's pose")
    lead_in, t_in = rise, len(rise) * DT
    lead_out, t_out = rise[::-1], len(rise) * DT
    hold = np.repeat(home[None], round(gestures.HOLD_S / DT) + 1, axis=0)
    q = np.concatenate([lead_in, q, lead_out, hold])
    pose, height = gestures.solve_resting_pose(robot, q)

    n = len(q)
    f32 = np.float32
    out = {"time": (np.arange(n) * DT).astype(f32), "dt": np.array(DT),
           "phase": np.ones(n, dtype=f32)}
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(dances._keys()):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array("tools/synth_rub_gesture.py (keyframes on bow.npz row "
                             f"{BOW_ROW})")
    out["lead_in_s"] = np.array(t_in)
    out["lead_out_s"] = np.array(t_out)
    out["hold_s"] = np.array(gestures.HOLD_S)
    dest = REPO / "tasks/jumper/gesture_rub/media/rub.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    vel = np.diff(q, axis=0) / DT
    print(f"rub -> {dest.relative_to(REPO)}: {n * DT:.2f} s "
          f"({t_in:.2f} in, {RUB_S:.2f} rub, {t_out:.2f} out, {gestures.HOLD_S:.2f} held)")
    print(f"  base z {pose[:, 2].min() * 1000:.0f}..{pose[:, 2].max() * 1000:.0f} mm, "
          f"tilt at most {np.degrees(np.abs(pose[:, 3:5]).max()):.1f} deg")
    print(f"  joint pos std median {np.median(q.std(0)):.3f} max {q.std(0).max():.3f}, "
          f"RMS amplitude {np.sqrt(((q - q.mean(0)) ** 2).mean()):.3f} rad")
    print(f"  joint vel RMS {np.sqrt((vel ** 2).mean()):.3f} rad/s, "
          f"peak {np.abs(vel).max():.2f}")
    print("  off the floor: " + ", ".join(
        f"{f} {v * 100:.0f}%" for f, v in
        zip(gestures.FEET, (height >= gestures.DOWN_TOL).mean(0))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
