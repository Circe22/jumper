"""jumper.gesture_peek -- hiding a giggle behind its claws, and peeking.

A one-shot gesture on the dance machinery: the clip `media/peek.npz` is synthesized
by `tools/synth_peek_gesture.py`. The robot rises as bow does, crosses both claws
in front of its face, shakes with giggles, opens the left claw to peek through and
snaps it shut, and sits back down. 11.7 s, starting and ending at `HOME`.

Idea by Circe for the crab-undercover short (shot 10). Export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_peek",
    assets=JUMPER_ASSETS,
    description="jumper hexapod hiding a giggle behind its claws and peeking; the clip "
                "is synthesized by tools/synth_peek_gesture.py into tasks/jumper/gesture_peek/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
