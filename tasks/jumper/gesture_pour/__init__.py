"""jumper.gesture_pour -- pouring vinegar into a dish.

A one-shot gesture on the dance machinery: the clip `media/pour.npz` is synthesized
by `tools/synth_pour_gesture.py`. The robot rises as bow does, holds its left claw upright like a bottle, tips it
over inward to pour, shakes it three times, rights it and sits back down. 11.5 s,
starting and ending at `HOME`.

For the crab-undercover short (shot 9), at Circe's request. Export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_pour",
    assets=JUMPER_ASSETS,
    description="jumper hexapod pouring vinegar with its left claw; the clip "
                "is synthesized by tools/synth_pour_gesture.py into tasks/jumper/gesture_pour/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
