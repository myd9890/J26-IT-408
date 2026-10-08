# Dataset Annotation Summary

The following tables summarize the annotation results for the three image classes:

- `banana_healthy_leaf`
- `black_sigatoka`
- `yellow_sigatoka`

Each class contains **20 images**, giving a total of **60 images**.

---

## 1. Usable for Scope

All images were considered suitable for the intended scope.

| Label | Yes |
|---|---:|
| `banana_healthy_leaf` | 20 |
| `black_sigatoka` | 20 |
| `yellow_sigatoka` | 20 |
| **Total** | **60** |

---

## 2. Background

The background was classified as either `field` or `plain`.

| Label | Field | Plain | Total |
|---|---:|---:|---:|
| `banana_healthy_leaf` | 1 | 19 | 20 |
| `black_sigatoka` | 15 | 5 | 20 |
| `yellow_sigatoka` | 12 | 8 | 20 |
| **Total** | **28** | **32** | **60** |

### Observation

- Most `banana_healthy_leaf` images have a `plain` background.
- Most `black_sigatoka` images have a `field` background.
- `yellow_sigatoka` images are also more frequently associated with a `field` background.

---

## 3. Leaves in Image

Images were classified according to whether they contained one main leaf or several mostly visible leaves.

| Label | 1 Leaf | Several Leaves | Total |
|---|---:|---:|---:|
| `banana_healthy_leaf` | 20 | 0 | 20 |
| `black_sigatoka` | 19 | 1 | 20 |
| `yellow_sigatoka` | 20 | 0 | 20 |
| **Total** | **59** | **1** | **60** |

### Observation

Almost all images contain **one main leaf**. Only one `black_sigatoka` image contains several mostly visible leaves.

---

## 4. Lesion Visibility

| Label | No | Yes | Total |
|---|---:|---:|---:|
| `banana_healthy_leaf` | 20 | 0 | 20 |
| `black_sigatoka` | 0 | 20 | 20 |
| `yellow_sigatoka` | 0 | 20 | 20 |
| **Total** | **20** | **40** | **60** |

### Observation

- All `banana_healthy_leaf` images have **no visible lesions**.
- All `black_sigatoka` images have **visible lesions**.
- All `yellow_sigatoka` images have **visible lesions**.
- No images were labelled `unsure`.

---

## 5. Other Damage

Other visible damage includes tears, dry or brown edges, insect holes, or dirt that could potentially be mistaken for lesions.

| Label | No | Yes | Total |
|---|---:|---:|---:|
| `banana_healthy_leaf` | 20 | 0 | 20 |
| `black_sigatoka` | 4 | 16 | 20 |
| `yellow_sigatoka` | 1 | 19 | 20 |
| **Total** | **25** | **35** | **60** |

### Observation

- None of the `banana_healthy_leaf` images contain other potentially confusing damage.
- `black_sigatoka` has other damage in **16 of 20 images**.
- `yellow_sigatoka` has other damage in **19 of 20 images**.

---

## Summary

| Attribute | `banana_healthy_leaf` | `black_sigatoka` | `yellow_sigatoka` |
|---|---:|---:|---:|
| Total images | 20 | 20 | 20 |
| Usable for scope | 20 | 20 | 20 |
| Field background | 1 | 15 | 12 |
| Plain background | 19 | 5 | 8 |
| One main leaf | 20 | 19 | 20 |
| Several leaves | 0 | 1 | 0 |
| Lesion visible | 0 | 20 | 20 |
| No lesion | 20 | 0 | 0 |
| Other damage | 0 | 16 | 19 |
| No other damage | 20 | 4 | 1 |

> **Note:** `lesion_size` and `lighting` are not included in this summary because no annotation results for those fields were provided.

# Data Limitations

- All **20 of 20** banana healthy leaf images are usable for the defined scope.
- All **20 of 20** black sigatoka images are usable for the defined scope.
- All **20 of 20** yellow sigatoka images are usable for the defined scope.
- **0 of 20** banana healthy leaf images have several leaves.
- **1 of 20** black sigatoka images has several leaves.
- **0 of 20** yellow sigatoka images have several leaves.
- **1 of 20** banana healthy leaf images has a field background, while **19 of 20** have a plain background.
- **15 of 20** black sigatoka images have a field background, while **5 of 20** have a plain background.
- **12 of 20** yellow sigatoka images have a field background, while **8 of 20** have a plain background.
- **20 of 20** banana healthy leaf images have no visible lesions.
- **20 of 20** black sigatoka images have visible lesions.
- **20 of 20** yellow sigatoka images have visible lesions.
- **0 of 20** banana healthy leaf images have other damage that could be mistaken for a lesion.
- **16 of 20** black sigatoka images have other damage that could be mistaken for a lesion.
- **19 of 20** yellow sigatoka images have other damage that could be mistaken for a lesion.

These observations show that the dataset contains some visual variation in background and other leaf damage, particularly for the black sigatoka and yellow sigatoka classes. However, all **60 of 60** images were considered usable for the defined scope.