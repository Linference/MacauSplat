# MacauSplat 0.6 + QVC — work in progress

User requested pausing validation and uploading the complete current source on 2026-10-07. This branch contains the OOOSplat 0.6.0 merge plus the existing MacauSplat annotation/target-selection work and the new QVC quality/track-aware training preparation.

## Implemented

- Pinned COLMAP 4.2.1 and OOOBrush ooo-v1.0.0 integration from upstream 0.6.
- Separate training dataset preparation using image focus, nearby camera poses, and track coverage. Removing a view cannot leave its observed points with fewer than two views. At least 30 views and 80% of input views are retained.
- Robust reprojection-error and track-support filtering of initialization points. If fewer than 100 points or 80% of original points survive, the original point set is retained.
- Verified native Mip rendering, full degree-three SH, and diagnostic held-out view evaluation for sufficiently large new training datasets.
- Existing projects resume their saved policy; the original inputs, calibrated reconstruction, and final PLY remain intact.
- Original annotation, seed editor, review queue and sparse-cloud preview interfaces retained.

## Validation at pause

- Frontend build and backend compile succeeded in the staging copy.
- 387 backend tests passed; 3 existing tests ignored; 2 Windows symlink-privilege fixtures not executed successfully on this host.
- Two frontend test expectations still require adapting to the 0.6 task interface. Final frontend verification was not completed.
- Official new Brush executed on the RTX 5070 Laptop GPU and exported a valid PLY in its four-iteration runtime smoke check. This is not a model-quality benchmark.
- Actual St Paul training-data audit: all 243 registered views retained; initialization points 104320 -> 104015. These are data-preparation results, not proof of improved final reconstruction accuracy.
- Full same-budget reconstruction A/B comparison and final application acceptance remain pending. No dramatic improvement percentage is claimed.

The quality logic is MacauSplat-specific engineering; 3DGS, COLMAP, Brush and Mip-Splatting remain their upstream technologies. This is a source snapshot, not a validated release.
