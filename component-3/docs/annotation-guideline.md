# Annotation Guideline

**Date:** 2026-10-12

## 1. Classes

Background, leaf, and lesion. Lesion polygons sit on top of the leaf polygon.

**Reason:** This allows the complete leaf area and diseased area to be identified separately.

## 2. Leaf Mask

The leaf mask includes the whole main leaf blade, including lesions and attached dried tissue. Include the midrib, but exclude the petiole, stem, torn-off gaps, and other leaves.

**Reason:** The complete main leaf blade is needed to calculate the affected area accurately.

## 3. Lesion

Lesions include discoloured tissue belonging to a spot, streak, or patch. Include brown, dark, and grey necrotic tissue. Include yellow halos only when clearly attached to a spot.

**Reason:** These regions represent visible disease symptoms and help define the affected area.

## 4. Dried Edges and Tips

Include dried edges and tips as lesions and tag them as `edge_drying`.

**Reason:** This maintains consistency with the earlier observation decision.

## 5. Tiny Specks

Skip specks smaller than approximately 5 pixels and record this as a known small bias.

**Reason:** Very small specks are difficult to annotate consistently.

## 6. Boundary Rule

Trace polygons along the visible boundary of the leaf or lesion.

**Reason:** Using visible boundaries helps maintain consistent annotations.

## 7. Time Box

Allow 4 minutes per image. Flag an image as `hard` if annotation takes more than 6 minutes.

**Reason:** This helps monitor annotation time and identify difficult images.

## 8. Blindness

Annotate images without looking at the v0 overlays.

**Reason:** Independent annotations are necessary for a fair evaluation of the baseline model.