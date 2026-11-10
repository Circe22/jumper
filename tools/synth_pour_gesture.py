#!/usr/bin/env python3
"""Synthesize the vinegar-pouring gesture from hand-placed keyframes.

    python tools/synth_pour_gesture.py

Writes `tasks/jumper/gesture_pour/media/pour.npz` in the layout
`tools/import_wbc_gestures.py` writes.

## The motion

For shot 9 of the crab-undercover short (Circe, 2026-11-10): pouring vinegar into a
dish. The robot rises as `jumper.gesture_bow` does and lifts its left claw up in
front of the face as if holding a bottle. It tips the claw over (the wrist pitch),
gives it three little shakes to get the last drops out, tips it back upright and
lowers the claw, then sits down the way it rose. The right claw swings aside, out of the way.

Both poses are solved, not typed: `HOLD_LEFT` is the left arm's J0..J3 holding the
claw upright in front of the left of the face (tip at about (0.135, 0.070, 0.172)
in the base frame), `POUR_LEFT` tips it over inward (tip at about (0.167, -0.018,
0.107)), both from `scipy.optimize.least_squares` with nothing in contact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import import_wbc_dances as dances
import import_wbc_gestures as gestures
import numpy as np
import synth_rub_gesture as rub

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DT = rub.DT

#: Left arm J0..J3: the claw held upright, and tipped over to pour.
HOLD_LEFT = np.array([-1.463, -3.141, -1.417, 0.368])
POUR_LEFT = np.array([-0.717, -2.311, -1.528, 0.042])
#: The right shoulder yaw swung outward by this much while the left claw pours, rad.
RIGHT_ASIDE = 0.45
#: Seconds: bow pose to the lifted pose; tipping over; the pour held; tipping back.
TO_LIFT_S = 0.9
TIP_S = 0.6
POUR_S = 2.4
#: Three shakes during the pour: (start, end) seconds into the pour; wrist
#: amplitude and rate.
SHAKES = ((0.6, 0.95), (1.15, 1.5), (1.7, 2.05))
SHAKE_AMP = 0.10
SHAKE_HZ = 7.0


def arm_pose(base: np.ndarray, left: np.ndarray) -> np.ndarray:
    keys = dances._keys()
    q = base.copy()
    for j in range(4):
        q[keys.index(f"L0_j{j}")] = left[j]
    # The right claw steps aside, out of the tipped claw's way.
    q[keys.index("L5_j0")] -= RIGHT_ASIDE
    return q


def build() -> tuple[np.ndarray, np.ndarray]:
    keys = dances._keys()
    col = {k: i for i, k in enumerate(keys)}
    rise = rub.bow_rise()
    b = rise[-1]
    lp = arm_pose(b, HOLD_LEFT)

    def ramp(a: np.ndarray, z: np.ndarray, s: float) -> np.ndarray:
        n = round(s / DT)
        u = gestures._smoothstep(np.arange(n) / n)[:, None]
        return a + (z - a) * u

    to_lift = ramp(b, lp, TO_LIFT_S)
    tipped = arm_pose(b, POUR_LEFT)
    tip_over = ramp(lp, tipped, TIP_S)

    t = np.arange(round(POUR_S / DT)) * DT
    pour = np.repeat(tipped[None], len(t), axis=0)
    shake = sum(rub._window(t, a, e, 0.05) for a, e in SHAKES)
    pour[:, col["L0_j1"]] += SHAKE_AMP * shake * np.sin(2 * np.pi * SHAKE_HZ * t)

    return rise, np.concatenate([to_lift, tip_over, pour, tip_over[::-1], to_lift[::-1]])


def main() -> int:
    from tasks.jumper.common.constants import HOME

    robot = gestures._Robot()
    rise, mid = build()
    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - mid.min(axis=0), mid.max(axis=0) - hi)
    for i in np.nonzero(over > 0)[0]:
        print(f"  clamped {robot.names[i]} by up to {over[i] * 1000:.1f} mrad")
    mid = np.clip(mid, lo, hi)
    home = np.array([HOME[n] for n in robot.names])
    hold = np.repeat(home[None], round(gestures.HOLD_S / DT) + 1, axis=0)
    q = np.concatenate([rise, mid, rise[::-1], hold])
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
    out["source"] = np.array("tools/synth_pour_gesture.py (keyframes; rise from bow.npz)")
    out["lead_in_s"] = np.array(len(rise) * DT)
    out["lead_out_s"] = np.array(len(rise) * DT)
    out["hold_s"] = np.array(gestures.HOLD_S)
    dest = REPO / "tasks/jumper/gesture_pour/media/pour.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    vel = np.diff(q, axis=0) / DT
    print(f"pour -> {dest.relative_to(REPO)}: {n * DT:.2f} s")
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
