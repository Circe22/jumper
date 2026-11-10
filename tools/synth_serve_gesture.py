#!/usr/bin/env python3
"""Synthesize the dish-serving gesture from hand-placed keyframes.

    python tools/synth_serve_gesture.py

Writes `tasks/jumper/gesture_serve/media/serve.npz` in the layout
`tools/import_wbc_gestures.py` writes.

## The motion

For shot 9 of the crab-undercover short (Circe, 2026-11-10): serving a little
dipping-sauce dish. The robot rises as `jumper.gesture_bow` does, brings both claws
together in front of it as if carrying a small dish, holds it level with a little
bob, offers it forward ("your sauce"), holds there, draws it back and sits down the
way it rose. The dish itself is added later in the film, not simulated.

Both poses are solved, not typed: `CARRY_LEFT` is the left arm's J0..J3 with the
claw tip at about (0.141, 0.027, 0.052) in the base frame, `SERVE_LEFT` reaches
forward to about (0.186, 0.028, 0.067); the right arm mirrors the left (J1 shared
sign, J0/J2/J3 flipped). Both from `scipy.optimize.least_squares` with nothing in
contact.
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

#: Left arm J0..J3: carrying the dish, and offering it forward.
CARRY_LEFT = np.array([-0.288, -2.219, -0.455, -1.579])
SERVE_LEFT = np.array([-0.267, -2.137, -0.986, -0.789])
#: Right arm sign per joint for the mirror.
MIRROR = np.array([-1.0, 1.0, -1.0, -1.0])
#: Seconds: bow pose to carrying; carrying with a bob; carrying to offering; the
#: offer held.
TO_CARRY_S = 0.9
CARRY_S = 1.6
TO_SERVE_S = 0.8
SERVE_S = 1.8
#: The bob while carrying: shoulder pitch amplitude, rad, and rate, Hz.
BOB_AMP = 0.05
BOB_HZ = 1.5


def arm_pose(base: np.ndarray, left: np.ndarray) -> np.ndarray:
    keys = dances._keys()
    q = base.copy()
    for j in range(4):
        q[keys.index(f"L0_j{j}")] = left[j]
        q[keys.index(f"L5_j{j}")] = MIRROR[j] * left[j]
    return q


def build() -> tuple[np.ndarray, np.ndarray]:
    keys = dances._keys()
    col = {k: i for i, k in enumerate(keys)}
    rise = rub.bow_rise()
    b = rise[-1]
    carry = arm_pose(b, CARRY_LEFT)
    serve = arm_pose(b, SERVE_LEFT)

    def ramp(a: np.ndarray, z: np.ndarray, s: float) -> np.ndarray:
        n = round(s / DT)
        u = gestures._smoothstep(np.arange(n) / n)[:, None]
        return a + (z - a) * u

    to_carry = ramp(b, carry, TO_CARRY_S)
    t = np.arange(round(CARRY_S / DT)) * DT
    held = np.repeat(carry[None], len(t), axis=0)
    bob = BOB_AMP * np.sin(2 * np.pi * BOB_HZ * t) * rub._window(t, 0.0, CARRY_S, 0.2)
    held[:, col["L0_j1"]] += bob
    held[:, col["L5_j1"]] += bob
    to_serve = ramp(carry, serve, TO_SERVE_S)
    offer = np.repeat(serve[None], round(SERVE_S / DT), axis=0)
    return rise, np.concatenate([to_carry, held, to_serve, offer, to_serve[::-1], to_carry[::-1]])


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
    out["source"] = np.array("tools/synth_serve_gesture.py (keyframes; rise from bow.npz)")
    out["lead_in_s"] = np.array(len(rise) * DT)
    out["lead_out_s"] = np.array(len(rise) * DT)
    out["hold_s"] = np.array(gestures.HOLD_S)
    dest = REPO / "tasks/jumper/gesture_serve/media/serve.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    vel = np.diff(q, axis=0) / DT
    print(f"serve -> {dest.relative_to(REPO)}: {n * DT:.2f} s")
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
