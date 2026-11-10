"""jumper.gesture_serve -- serving a little dipping-sauce dish.

A one-shot gesture on the dance machinery: the clip `media/serve.npz` is synthesized
by `tools/synth_serve_gesture.py`. The robot rises as bow does, carries an imagined dish level in both claws with a
little bob, offers it forward, holds, draws it back and sits down. 12.9 s,
starting and ending at `HOME`.

For the crab-undercover short (shot 9), at Circe's request. Export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_serve",
    assets=JUMPER_ASSETS,
    description="jumper hexapod serving a dish with both claws; the clip "
                "is synthesized by tools/synth_serve_gesture.py into tasks/jumper/gesture_serve/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
