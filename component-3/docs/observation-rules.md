# Image Annotation Guidelines

Use the following labels to describe the visible characteristics of each image.

---

## 1. Background

Describe what is visible behind the main leaf.

| Label | Description |
|---|---|
| `plain` | One uniform colour; nothing else is visible. |
| `field` | Soil, other plants, other leaves, or sky is visible. |
| `hand` | A hand or fingers are touching the leaf. |
| `other` | Anything else is visible, such as a table, paper, or clutter. |

---

## 2. Leaves in Image

Count the number of main leaves that are mostly visible.

| Label | Description |
|---|---|
| `1` | One main leaf is visible. |
| `several` | Two or more leaves are mostly visible. |

---

## 3. Lesion Visible

Determine whether a visible lesion can be identified on the main leaf.

| Label | Description |
|---|---|
| `yes` | You can clearly point to a lesion. |
| `unsure` | It could be a lesion or natural ageing, or you would need to zoom in to decide. |
| `no` | No lesion is visible. |

---

## 4. Lesion Size

Estimate the share of the **main leaf** covered by the lesion(s) by eye.

| Label | Description |
|---|---|
| `small` | Less than **5%** of the main leaf is covered. |
| `medium` | **5–25%** of the main leaf is covered. |
| `large` | More than **25%** of the main leaf is covered. |
| `na` | Use when no lesion is visible. |

> **Note:** These size bins are for observation only. They are **not** the severity bands that will be defined later.

---

## 5. Lighting

Describe the lighting conditions that affect how clearly the leaf and lesions can be seen.

| Label | Description |
|---|---|
| `good` | Lesion edges are clear and easy to see. |
| `dark` | Details are lost in shadow. |
| `glare` | Bright reflections wash out parts of the image. |

---

## 6. Other Damage

Check whether there is other visible damage that could be mistaken for a lesion.

| Label | Description |
|---|---|
| `yes` | Tears, dry or brown edges, insect holes, or dirt are present and could be mistaken for a lesion. |
| `no` | No such damage is visible. |

---

## 7. Usable for Scope

Decide whether the image can be reliably used for the intended analysis.

| Label | Description |
|---|---|
| `yes` | You can draw a clean outline around the leaf and around **every visible lesion**. For healthy leaves, use `yes` if the leaf outline is clean. |
| `no` | The leaf or lesions cannot be outlined cleanly, or the image is too unclear to annotate reliably. |

> **Important:** When in doubt, choose `no`.

---

## Quick Reference

| Field | Allowed Values |
|---|---|
| `background` | `plain`, `field`, `hand`, `other` |
| `leaves_in_image` | `1`, `several` |
| `lesion_visible` | `yes`, `unsure`, `no` |
| `lesion_size` | `small`, `medium`, `large`, `na` |
| `lighting` | `good`, `dark`, `glare` |
| `other_damage` | `yes`, `no` |
| `usable_for_scope` | `yes`, `no` |