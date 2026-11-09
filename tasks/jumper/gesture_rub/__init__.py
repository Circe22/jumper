"""jumper.gesture_rub -- rubbing the claws together, a villain's snicker.

A one-shot gesture on the dance machinery, like `jumper.gesture_paw`: the clip
`media/rub.npz` is synthesized by `tools/synth_rub_gesture.py` from bow's
claws-together pose. The robot raises both claws in front of its chest, rubs them
in antiphase for three seconds with the jaws chattering, and its body shakes three
times. 9.1 s, starting and ending at `HOME`.

Made for the crab-undercover short (shot 10). Export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_rub",
    assets=JUMPER_ASSETS,
    description="jumper hexapod rubbing its claws together (a snicker); the clip is "
                "synthesized by tools/synth_rub_gesture.py into tasks/jumper/gesture_rub/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
