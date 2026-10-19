**Date:** 2026-10-12

**Rule:** Baseline v0 is insufficient if, on my hand-traced test masks, median lesion Dice < 0.70 OR Scope MAE > 10 percentage points.

**Justification:** A median lesion Dice below 0.70 indicates insufficient overlap between the predicted lesion masks and the manually annotated masks. A Scope MAE above 10 percentage points could move a plant into an incorrect severity band.

**Note:** I had already seen visual QA of v0 before writing this; I had not seen any mask-based metric.