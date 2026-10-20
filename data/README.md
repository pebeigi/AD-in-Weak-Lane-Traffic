# Jounieh weak-lane freeway trajectory dataset

## Summary

`Final_Lebanon_Data.xz` is an XZ-compressed CSV containing 3,006,441 sampled
positions from 7,486 vehicle trajectories. The trajectories come from two
drone recordings of both directions of a congested freeway section in
Jounieh, Lebanon, collected during July and August 2023. Samples are spaced
at 0.1 s. The canonical trajectory key is `(run_id, lane_kf, id)`; `id` alone
is not globally unique.

The file supports the dataset characterization, weak-lane interaction
measurements, and paired SDPM--RAU model comparison reported in the
accompanying manuscript. The original video and intermediate detector,
tracker, and homography files are not part of this release.

## File and integrity information

| File | Contents | Size | SHA-256 |
|---|---|---:|---|
| `Final_Lebanon_Data.xz` | XZ-compressed CSV | 65,586,856 bytes | `0a949b5c9d93f4d4ee95a3b27fdd36e1dadd08506c5cb7461c4a6043f206ddbf` |
| `Final_Lebanon_Data.csv` | Uncompressed CSV produced locally | 294,731,898 bytes | `5807e3122a15b89dcf19e2ec13fa8d8126466267df711361801edfc5cbf9268a` |

Run `python -m analysis.verify_data --extract` from the repository root to
verify both digests and create the uncompressed CSV. The command refuses to
overwrite an existing file whose checksum differs.

## Variables and units

| Column | Definition | Unit or values | Missing values |
|---|---|---|---:|
| `id` | Tracker-assigned vehicle identifier; unique only within a recording and direction | Positive integer | 0 |
| `time` | Time within the processed recording | s | 0 |
| `xloc_kf` | Smoothed vehicle-center x coordinate in the site-local ground plane | m | 0 |
| `yloc_kf` | Smoothed vehicle-center y coordinate in the site-local ground plane | m | 0 |
| `lane_kf` | Travel-direction code, despite the historical column name | `1`, `2` | 0 |
| `speed_kf` | Smoothed scalar speed | m s\(^{-1}\) | 0 |
| `acceleration_kf` | Smoothed scalar acceleration | m s\(^{-2}\) | 0 |
| `length_smoothed` | Smoothed estimated vehicle length | m | 6,142 |
| `width_smoothed` | Smoothed estimated vehicle width | m | 6,142 |
| `class` | Inferred vehicle class: `0` short/motorcycle, `1` long/heavy vehicle, `2` light vehicle | Categorical code | 0 |
| `run_id` | Recording identifier | `1`, `2` | 0 |

The class meanings were inferred from the stored vehicle dimensions and the
three annotation classes; the numeric labels are not embedded in the source
file. The coordinates are metric, site-local coordinates. A formal coordinate
reference system and its origin were not preserved with the processed data.

## Methods and provenance

Footage was recorded from a DJI Mini 3 at 30 frames per second in 4K. A
YOLOv8 detector and DeepSORT tracker produced vehicle detections and tracks.
Twenty fixed control points were used for planar homography-based
georeferencing, after which positions, speeds, accelerations, and vehicle
dimensions were smoothed. The detector/tracker settings, homography residuals,
exact flight dates and times, flight height, and camera angle were not
preserved. See the manuscript and repository README for the full scope and
limitations.

## Software and outputs

The repository root contains the analysis and model code. `outputs/` contains
the frozen cohort, calibration results, statistics, tables, and figures.
`outputs/statistics/environment.txt` records the original analysis environment,
and `requirements-lock.txt` provides exact Python package versions.

## Access and license

The dataset is public in this repository and licensed under the Creative
Commons Attribution 4.0 International License; see `LICENSE` in this directory.
The analysis software is separately licensed under the MIT License.

## Citation

Until a versioned archive DOI is minted, cite the accompanying manuscript and
identify the repository and release version used. Dataset-specific citation
metadata are in this directory's `CITATION.cff`; software and paper citation
metadata are in the repository-root `CITATION.cff`. Replace the repository URL
with the dataset DOI in publications once the archived record is public.
