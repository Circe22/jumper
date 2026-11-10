# jumper.gesture_serve — trained on Colab, 2026-11-10

Serving a little dish with both claws (crab-undercover short, shot 9).
Reference clip: `tools/synth_serve_gesture.py` → `../../media/serve.npz` (12.9 s).

Trained on a Colab T4 with `notebooks/jumper_colab.ipynb` from branch `crab-peek`
(commit 4bba250): 256 environments, 1000 updates, about 20 minutes.

| File | Contents |
|---|---|
| `model_999.pt` | rsl_rl checkpoint after the last update (actor, critic, optimizer) |
| `actor.onnx` | exported actor, `scripts/export.py --no-video` |
| `layout.json` | the deployment contract written alongside it by `scripts/export.py` |

## Training, measured

Success is episodes ending on `time_out` over all episode ends (the other ends are
the reference-tracking terminations, mostly `ee_body_pos`: a claw too far from the
reference). Averaged over ±10 updates:

| Update | 0–250 | 300 | 350 | 400 | 600 | 850+ |
|---|---|---|---|---|---|---|
| Success | 0% | 1% | 71% | 87% | 97% | 100% |

Episodes reach the full 320 steps (6.4 s) from about update 400; the policy std
falls from 1.0 to 0.21. This is simulation only: no physical-robot check yet.
