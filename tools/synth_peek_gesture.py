#!/usr/bin/env python3
"""Synthesize the giggle-and-peek gesture from hand-placed keyframes.

    python tools/synth_peek_gesture.py

Writes `tasks/jumper/gesture_peek/media/peek.npz` in the layout
`tools/import_wbc_gestures.py` writes.

## The motion

Idea by the director (Circe, 2026-11-09): "put both claws under the face and shake,
like trying to hide a laugh behind your hands but the eyes can't help peeking."

The robot rises as `jumper.gesture_bow` does, brings both claw tips in under its
face (the display), and giggles: the body shakes in bursts with the claws trembling.
In the middle the left claw swings out to open a gap -- the face's eyes peek through
it on the screen -- holds, and snaps back. Then it lowers its claws and sits down
the way it rose.

The cupped pose is solved, not typed: `CUP_LEFT` is the left arm's J0..J3 that puts
the claw tip at `CUP_TIP` in the base frame (just in front of the display's lower
edge) with nothing in contact, from `scipy.optimize.least_squares`; the right arm is
its mirror. The legs are bow's at `synth_rub_gesture.BOW_ROW`.
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

#: Left arm J0..J3 with the claw tip at CUP_TIP; right arm is the mirror.
CUP_LEFT = np.array([-0.489, -2.298, -0.258, -1.819])
CUP_TIP = (0.118, 0.016, 0.050)

#: Each shoulder yaw swung in by this much from the cheek pose, rad: the claws
#: cover the face.
COVER = 0.34

#: Seconds: bow-row pose to the cup, and the cup back to it.
TO_CUP_S = 0.8

#: The giggle: total seconds held at the cup.
GIGGLE_S = 4.0
#: Shake bursts (start, end), seconds into the giggle; knee bob amplitude and rate.
SHAKE_BURSTS = ((0.2, 0.9), (1.1, 1.7), (2.9, 3.7))
SHAKE_AMP = 0.07
SHAKE_HZ = 6.0
#: Claws tremble with the shake: shoulder-pitch amplitude, rad.
TREMBLE_AMP = 0.035
#: The peek: the left shoulder yaw opens by PEEK_OPEN (back out), easing out over
#: PEEK_T[0]..PEEK_T[1], held to PEEK_T[2], snapped back by PEEK_T[3].
PEEK_OPEN = 0.38
PEEK_T = (1.85, 2.15, 2.55, 2.70)


def _ease(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return gestures._smoothstep(np.clip((t - a) / (b - a), 0.0, 1.0))


def cup_pose(base: np.ndarray) -> np.ndarray:
    keys = dances._keys()
    q = base.copy()
    for j in range(4):
        q[keys.index(f"L0_j{j}")] = CUP_LEFT[j]
    # Mirror: J1 shared sign, J0/J2/J3 flipped.
    for j, s in ((0, -1), (1, 1), (2, -1), (3, -1)):
        q[keys.index(f"L5_j{j}")] = s * CUP_LEFT[j]
    q[keys.index("L0_j4")] = 0.0
    q[keys.index("L5_j4")] = 0.0
    # Swing both claws in across the face from the cheeks: covering it.
    q[keys.index("L0_j0")] -= COVER
    q[keys.index("L5_j0")] += COVER
    return q


def build() -> tuple[np.ndarray, np.ndarray]:
    """(the rise, the whole clip after the rise up to the sit-down)."""
    keys = dances._keys()
    col = {k: i for i, k in enumerate(keys)}
    rise = rub.bow_rise()
    b = rise[-1]
    c = cup_pose(b)

    n_to = round(TO_CUP_S / DT)
    u = gestures._smoothstep(np.arange(n_to) / n_to)[:, None]
    to_cup = b + (c - b) * u

    t = np.arange(round(GIGGLE_S / DT)) * DT
    g = np.repeat(c[None], len(t), axis=0)
    shake = sum(rub._window(t, a, e, 0.08) for a, e in SHAKE_BURSTS)
    wave = np.sin(2 * np.pi * SHAKE_HZ * t)
    for leg, sign in (("L1", 1), ("L2", 1), ("L3", -1), ("L4", -1)):
        g[:, col[f"{leg}_j1"]] += sign * SHAKE_AMP * shake * wave
    g[:, col["L0_j1"]] += TREMBLE_AMP * shake * wave
    g[:, col["L5_j1"]] += TREMBLE_AMP * shake * wave
    a0, a1, a2, a3 = PEEK_T
    peek = _ease(t, a0, a1) * (1.0 - _ease(t, a2, a3))
    # The left claw swings back out to the cheek: a gap the eyes look through.
    g[:, col["L0_j0"]] += PEEK_OPEN * peek

    back = to_cup[::-1]
    return rise, np.concatenate([to_cup, g, back])


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
    out["source"] = np.array("tools/synth_peek_gesture.py (keyframes; rise from bow.npz)")
    out["lead_in_s"] = np.array(len(rise) * DT)
    out["lead_out_s"] = np.array(len(rise) * DT)
    out["hold_s"] = np.array(gestures.HOLD_S)
    dest = REPO / "tasks/jumper/gesture_peek/media/peek.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    vel = np.diff(q, axis=0) / DT
    print(f"peek -> {dest.relative_to(REPO)}: {n * DT:.2f} s")
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
