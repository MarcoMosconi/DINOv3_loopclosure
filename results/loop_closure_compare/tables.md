### Retrieval, final layer, appearance only

| descriptor | model | R@1 | R@5 | AP | R@100P |
|---|---|---|---|---|---|
| ORB BoW | ORB | 0.836 | 0.982 | 0.731 | 0.345 |
| cls | DINOv2 | 0.545 | 0.818 | 0.414 | 0.200 |
| cls | DINOv3 | 0.491 | 0.782 | 0.335 | 0.036 |
| cls+mean | DINOv2 | 0.691 | 0.927 | 0.550 | 0.236 |
| cls+mean | DINOv3 | 0.527 | 0.800 | 0.400 | 0.109 |
| gem | DINOv2 | 0.764 | 1.000 | 0.591 | 0.255 |
| gem | DINOv3 | 0.855 | 1.000 | 0.733 | 0.255 |
| mean | DINOv2 | 0.545 | 1.000 | 0.427 | 0.291 |
| mean | DINOv3 | 0.873 | 1.000 | 0.750 | 0.218 |
| vlad | DINOv2 | 0.654 | 0.982 | 0.574 | 0.327 |
| vlad | DINOv3 | 0.800 | 0.964 | 0.684 | 0.200 |

### Layer x scale ablation, AP

| layer /  | scale /  | cls / DINOv2 | cls / DINOv3 | gem / DINOv2 | gem / DINOv3 | mean / DINOv2 | mean / DINOv3 | vlad / DINOv2 | vlad / DINOv3 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.000 | 0.629 | 0.610 | 0.782 | 0.639 | 0.581 | 0.560 | 0.709 | 0.615 |
| 9 | 1.000 | 0.531 | 0.686 | 0.805 | 0.894 | 0.821 | 0.879 | 0.645 | 0.723 |
| 12 | 0.500 | 0.432 | 0.312 | 0.551 | 0.831 | 0.430 | 0.774 | 0.598 | 0.648 |
| 12 | 1.000 | 0.414 | 0.335 | 0.591 | 0.733 | 0.427 | 0.750 | 0.574 | 0.684 |

### Layer x scale ablation, R@100P

| layer /  | scale /  | cls / DINOv2 | cls / DINOv3 | gem / DINOv2 | gem / DINOv3 | mean / DINOv2 | mean / DINOv3 | vlad / DINOv2 | vlad / DINOv3 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.000 | 0.345 | 0.036 | 0.473 | 0.436 | 0.200 | 0.109 | 0.309 | 0.218 |
| 9 | 1.000 | 0.109 | 0.145 | 0.473 | 0.527 | 0.491 | 0.564 | 0.309 | 0.545 |
| 12 | 0.500 | 0.145 | 0.018 | 0.255 | 0.255 | 0.200 | 0.236 | 0.236 | 0.109 |
| 12 | 1.000 | 0.200 | 0.036 | 0.255 | 0.255 | 0.291 | 0.218 | 0.327 | 0.200 |

### Best retrieval configuration per model

| model | layer | scale | desc | R@1 | R@5 | AP | R@100P |
|---|---|---|---|---|---|---|---|
| DINOv3 | 9 | 1.000 | gem | 0.946 | 1.000 | 0.894 | 0.527 |
| DINOv2 | 9 | 1.000 | mean | 0.873 | 1.000 | 0.821 | 0.491 |

### All-block sweep, AP

| layer /  | cls / DINOv3 | mean / DINOv3 | gem / DINOv3 | cls / DINOv2 | mean / DINOv2 | gem / DINOv2 |
|---|---|---|---|---|---|---|
| 1 | 0.002 | 0.152 | 0.158 | 0.202 | 0.164 | 0.476 |
| 2 | 0.322 | 0.100 | 0.283 | 0.182 | 0.251 | 0.566 |
| 3 | 0.209 | 0.167 | 0.386 | 0.365 | 0.355 | 0.556 |
| 4 | 0.318 | 0.316 | 0.459 | 0.488 | 0.550 | 0.602 |
| 5 | 0.531 | 0.448 | 0.493 | 0.536 | 0.528 | 0.722 |
| 6 | 0.610 | 0.560 | 0.639 | 0.629 | 0.581 | 0.782 |
| 7 | 0.542 | 0.645 | 0.760 | 0.580 | 0.570 | 0.726 |
| 8 | 0.592 | 0.794 | 0.815 | 0.514 | 0.666 | 0.748 |
| 9 | 0.686 | 0.879 | 0.894 | 0.531 | 0.821 | 0.804 |
| 10 | 0.803 | 0.843 | 0.827 | 0.530 | 0.824 | 0.817 |
| 11 | 0.344 | 0.776 | 0.741 | 0.552 | 0.690 | 0.725 |
| 12 | 0.304 | 0.779 | 0.656 | 0.399 | 0.409 | 0.595 |
| final | 0.335 | 0.750 | 0.733 | 0.414 | 0.427 | 0.591 |

### All-block sweep, R@100P

| layer /  | cls / DINOv3 | mean / DINOv3 | gem / DINOv3 | cls / DINOv2 | mean / DINOv2 | gem / DINOv2 |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 0.073 | 0.018 | 0.073 | 0.018 | 0.145 |
| 2 | 0.018 | 0.018 | 0.200 | 0.018 | 0.073 | 0.200 |
| 3 | 0.036 | 0.018 | 0.145 | 0.054 | 0.109 | 0.345 |
| 4 | 0.073 | 0.036 | 0.054 | 0.182 | 0.127 | 0.400 |
| 5 | 0.054 | 0.127 | 0.091 | 0.200 | 0.109 | 0.291 |
| 6 | 0.036 | 0.109 | 0.436 | 0.345 | 0.200 | 0.473 |
| 7 | 0.054 | 0.200 | 0.473 | 0.255 | 0.182 | 0.200 |
| 8 | 0.091 | 0.291 | 0.364 | 0.255 | 0.345 | 0.382 |
| 9 | 0.145 | 0.564 | 0.527 | 0.109 | 0.491 | 0.473 |
| 10 | 0.345 | 0.473 | 0.309 | 0.109 | 0.418 | 0.327 |
| 11 | 0.000 | 0.364 | 0.236 | 0.164 | 0.145 | 0.236 |
| 12 | 0.036 | 0.218 | 0.182 | 0.164 | 0.291 | 0.309 |
| final | 0.036 | 0.218 | 0.255 | 0.200 | 0.291 | 0.255 |

### Retrieval + top-5 geometric verification

| pipeline | model | R@1 | R@5 | AP | R@100P |
|---|---|---|---|---|---|
| L12 mean + patch verif. | DINOv2 | 0.873 | 1.000 | 0.836 | 0.491 |
| L12 mean + patch verif. | DINOv3 | 0.946 | 1.000 | 0.934 | 0.673 |
| L9 GeM + ORB verif. | DINOv2 | 0.964 | 1.000 | 0.907 | 0.345 |
| L9 GeM + ORB verif. | DINOv3 | 0.982 | 1.000 | 0.937 | 0.345 |
| L9 GeM + patch verif. | DINOv2 | 0.946 | 1.000 | 0.938 | 0.855 |
| L9 GeM + patch verif. | DINOv3 | 0.964 | 1.000 | 0.957 | 0.764 |
| ORB BoW + ORB verif. | ORB | 0.927 | 0.982 | 0.828 | 0.400 |

### Relative-pose accuracy of verified true loops

| matcher | model | pairs | median_trans_cm | p90_trans_cm | median_rot_deg | within_5cm_5deg | median_inliers |
|---|---|---|---|---|---|---|---|
| ORB | DINOv2 | 52.000 | 1.047 | 2.197 | 0.438 | 0.981 | 473.000 |
| ORB | DINOv3 | 52.000 | 0.971 | 2.248 | 0.431 | 1.000 | 512.000 |
| patch + LK | DINOv2 | 52.000 | 1.188 | 2.339 | 0.491 | 1.000 | 101.500 |
| patch + LK | DINOv3 | 52.000 | 1.032 | 3.069 | 0.447 | 0.981 | 88.500 |
| patch coarse | DINOv2 | 52.000 | 3.370 | 7.888 | 1.110 | 0.750 | 173.500 |
| patch coarse | DINOv3 | 52.000 | 4.093 | 6.087 | 1.157 | 0.712 | 234.500 |
| patch softargmax | DINOv2 | 52.000 | 4.583 | 9.067 | 1.422 | 0.577 | 68.000 |
| patch softargmax | DINOv3 | 52.000 | 5.356 | 9.858 | 1.735 | 0.442 | 97.500 |

### Final pipeline: accepted loops by class (inlier ratio)

| tau | model | accepted | true loop | valid overlap | right place, bad pose | wrong place | recall_true_loops |
|---|---|---|---|---|---|---|---|
| 0.1 | DINOv2 | 89 | 53 | 31 | 5 | 0 | 0.964 |
| 0.1 | DINOv3 | 136 | 53 | 38 | 36 | 9 | 0.964 |
| 0.15 | DINOv2 | 70 | 53 | 16 | 1 | 0 | 0.964 |
| 0.15 | DINOv3 | 104 | 53 | 37 | 12 | 2 | 0.964 |
| 0.2 | DINOv2 | 57 | 51 | 6 | 0 | 0 | 0.927 |
| 0.2 | DINOv3 | 83 | 53 | 28 | 2 | 0 | 0.964 |
| 0.25 | DINOv2 | 45 | 45 | 0 | 0 | 0 | 0.818 |
| 0.25 | DINOv3 | 69 | 53 | 16 | 0 | 0 | 0.964 |
| 0.3 | DINOv2 | 31 | 31 | 0 | 0 | 0 | 0.564 |
| 0.3 | DINOv3 | 59 | 53 | 6 | 0 | 0 | 0.964 |
| 0.35 | DINOv2 | 12 | 12 | 0 | 0 | 0 | 0.218 |
| 0.35 | DINOv3 | 53 | 52 | 1 | 0 | 0 | 0.946 |

### Final pipeline: accepted loops by class (#inliers)

| tau | model | accepted | true loop | valid overlap | right place, bad pose | wrong place | recall_true_loops |
|---|---|---|---|---|---|---|---|
| 15 | DINOv2 | 106 | 53 | 38 | 15 | 0 | 0.964 |
| 15 | DINOv3 | 155 | 53 | 39 | 48 | 15 | 0.964 |
| 20 | DINOv2 | 99 | 53 | 35 | 11 | 0 | 0.964 |
| 20 | DINOv3 | 120 | 53 | 38 | 25 | 4 | 0.964 |
| 30 | DINOv2 | 91 | 53 | 32 | 6 | 0 | 0.964 |
| 30 | DINOv3 | 103 | 53 | 38 | 12 | 0 | 0.964 |
| 50 | DINOv2 | 80 | 53 | 26 | 1 | 0 | 0.964 |
| 50 | DINOv3 | 91 | 53 | 35 | 3 | 0 | 0.964 |
| 75 | DINOv2 | 67 | 53 | 14 | 0 | 0 | 0.964 |
| 75 | DINOv3 | 76 | 53 | 23 | 0 | 0 | 0.964 |
| 100 | DINOv2 | 56 | 50 | 6 | 0 | 0 | 0.909 |
| 100 | DINOv3 | 66 | 53 | 13 | 0 | 0 | 0.964 |

### Final pipeline (LK pose): separation by inlier ratio

| model | loosest safe ratio | true loops kept | extra valid overlaps kept | max ratio of a wrong place | max ratio of a bad pose | min ratio of a true loop | median ratio of a true loop | #bad top-1 (>=15 inl.) |
|---|---|---|---|---|---|---|---|---|
| DINOv3 | 0.250 | 53 | 16 | 0.182 | 0.224 | 0.335 | 0.431 | 63 |
| DINOv2 | 0.200 | 51 | 6 |  | 0.151 | 0.166 | 0.309 | 15 |

### Follow-up (pose from verifier): min #inliers with zero harmful loops

| pipeline | model | min_inliers_zero_harmful | true_loops_kept | valid_overlaps_kept |
|---|---|---|---|---|
| L12 mean + patch verif. | DINOv2 | 78 | 48 | 14 |
| L12 mean + patch verif. | DINOv3 | 93 | 52 | 13 |
| L9 GeM + ORB verif. | DINOv2 | 82 | 53 | 23 |
| L9 GeM + ORB verif. | DINOv3 | 101 | 54 | 23 |
| L9 GeM + patch verif. | DINOv2 | 78 | 53 | 12 |
| L9 GeM + patch verif. | DINOv3 | 117 | 53 | 11 |
| ORB BoW + ORB verif. | ORB | 77 | 51 | 33 |

### Feature extraction time (multi-layer pass, 8 CPU threads)

| input | model | ms/frame (CPU) |
|---|---|---|
| scale 0.5 | DINOv3 | 188 |
| scale 1.0 | DINOv2 | 1104 |
| scale 0.5 | DINOv2 | 212 |
| scale 1.0, all 12 blocks | DINOv3 | 960 |
| scale 1.0, all 12 blocks | DINOv2 | 1183 |
