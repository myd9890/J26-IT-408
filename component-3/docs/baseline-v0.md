## Baseline V0 – Initial Issues Identified from Metadata

An initial review of the 30-image metadata identified several potential issues that require further investigation:

- **Non-zero measurements in healthy leaves:** Several healthy-leaf images have non-zero `scope_v0` values. If this metric represents lesion area, these values may indicate false-positive lesion detection.
- **Variation in leaf fraction:** The `leaf_frac` values vary considerably across images. This may reflect differences in image composition or possible leaf segmentation inconsistencies.
- **Presence of other damage:** Several diseased-leaf images are marked with `other_damage=yes`. Additional damage may affect the accuracy of lesion detection and measurement.
- **Differences between lesion categories and numerical measurements:** Some images labelled with large lesions have comparatively low `scope_v0` values. The relationship between the qualitative lesion labels and numerical measurements should be investigated.
- **Dark lighting conditions:** Images captured under dark lighting may affect the reliability of leaf segmentation and lesion detection.

These observations are preliminary and are based on the supplied metadata. The actual images and Baseline V0 segmentation outputs must be inspected to confirm the failure modes and count how many of the 30 images are affected. The findings will help determine whether a U-Net-based segmentation approach is justified.