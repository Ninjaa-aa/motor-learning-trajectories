# Human motor learning: beyond completion time

## Question and relevance

What do movement smoothness and trajectory shape reveal about practice that duration and task accuracy miss? This project directly addresses CaCTüS Project 9, Learning to Move: Computational Analysis of Human Motor Skills, through behavioural trajectory analysis, numerical signal processing, representation learning and baseline comparisons.

## Data and contribution

Reanalysis of a public wrist-controlled cursor-steering experiment by Vaynee Sungeelee, Antoine Loriette, Olivier Sigaud and Baptiste Caramiaux. The 36 participants performed an arc-tracing task over two days under three practice schedules. This project did not collect the recordings and does not claim to reproduce every result of the authors' paper. It adds a participant-disjoint comparison of phase classifiers, explicit input auditing, a dimensionless smoothness measure, and preprocessing sensitivity checks.

- Dataset version: https://doi.org/10.5281/zenodo.7824331 (CC BY 4.0).
- Primary paper: https://doi.org/10.1038/s41598-024-53253-3 .
- Attribution and licence: https://creativecommons.org/licenses/by/4.0/ .
- Modifications: parsing the original NeDB logs, movement segmentation, resampling, smoothing, feature extraction and analysis. Derived outputs omit raw wall-clock timestamps and generated source usernames.

Four participants (P1-EA, P1-RM, P9-CL, P12-EA) were reserved for development. The remaining 32 participants enter the reported descriptive evaluation; 32 satisfy the minimum four valid trials per class for classification. Across development and evaluation, 1573 matched-task trials were included and 155 were excluded, with every reason recorded. Development participants enter training folds only, never the reported test scores or paired changes.

## Input and identity checks

Participant identity uses the full filename, including condition: P1-CL and P1-EA are different people. The source describes the numeric portion as an ID but repeats numbers across conditions. Three full IDs (P11-EA, P5-CL, P8-RM) have differing generated usernames in their day-1/day-2 configurations. Their within-day pre/post comparisons remain valid; their paired retention comparisons are conservatively omitted because identity cannot be independently resolved. Filename-based retention identity is otherwise assumed from the documented archive structure.

The downloader checks the published archive size and MD5 and records SHA-256. Each raw file is hashed. The parser replays NeDB document updates and deletions, removes identical duplicate sample timestamps, and rejects conflicting timestamps. It never treats repeated document versions as extra observations.

## Fixed method

Only diameter 800 px, width 18 px and instructed duration 1 s are used. Day-1 blocks 1–3 are pre-test; 95–97 are post-test. Day-2 blocks 0–1 are retention. Other task geometries and training blocks are excluded by design, preventing task difficulty from becoming a phase cue.

Coordinates are screen pixels projected from wrist motion, not calibrated 3D hand positions. Based on the observed task geometry, movement starts at the last sample within 30 px of (100,500) before first reaching within 30 px of (900,500). This operational boundary rule removes waiting and repositioning; it is not a manually annotated physiological onset. Segments require at least 20 samples, duration 0.2–5 s, and no sample gap above 150 ms. All exclusions are retained. This can preferentially omit poor movements and is a limitation.

The camera timestamp `ts` is used in seconds. Segments are resampled at 120 Hz and smoothed with a cubic Savitzky–Golay window of about 125 ms. The study also checks 83 ms and 167 ms windows. Smoothing choices were fixed before evaluation; they are not selected by the best result.

Features are segmented duration, the source software's in-channel fraction, path-length/displacement ratio, radial RMS deviation divided by channel width, and log dimensionless jerk:

`LDJ = -log(T^5 / L^2 * integral(||d^3 x / dt^3||^2 dt))`

Here T is movement duration and L is smoothed path length. Higher LDJ means smoother movement under this definition. This differs from the source article's reported jerk measure, so numerical values and tests are not claimed as an exact replication. Normalised trajectory variability is the RMS spread around a participant's phase-mean trajectory.

For shape representation, each trajectory is resampled to 64 equal-time points, translated to its start, rotated toward its endpoint and divided by endpoint displacement. This retains shape and within-movement timing, while removing screen translation, orientation and displacement scale. PCA reduces the 128 coordinates to five components.

## Participant-level changes

Each participant contributes a phase mean, regardless of their number of valid trials. Differences use paired participants; bootstrap intervals resample participants 5,000 times. Wilcoxon signed-rank tests receive Holm correction across the 12 reported comparisons. These are exploratory secondary analyses on an existing dataset whose original findings were already known.

| Measure | Comparison | Participants | Mean change | 95% interval | Holm-adjusted p |
|---|---|---:|---:|---|---:|
| duration | post minus pre | 32 | 0.0038 | -0.3021 to 0.2887 | 0.5239 |
| duration | retention minus pre | 29 | -0.2814 | -0.5561 to -0.0371 | 0.241 |
| icf | post minus pre | 32 | 0.1667 | 0.1196 to 0.2121 | 4.48e-06 |
| icf | retention minus pre | 29 | 0.0562 | 0.0080 to 0.1031 | 0.09678 |
| ldj | post minus pre | 32 | 0.6857 | 0.0381 to 1.3422 | 0.241 |
| ldj | retention minus pre | 29 | 0.8415 | 0.2808 to 1.4600 | 0.1305 |
| path_ratio | post minus pre | 32 | -0.0995 | -0.1353 to -0.0665 | 5.635e-07 |
| path_ratio | retention minus pre | 29 | -0.0444 | -0.0707 to -0.0175 | 0.04453 |
| radial_rmse | post minus pre | 32 | -0.6798 | -0.8518 to -0.5091 | 3.912e-07 |
| radial_rmse | retention minus pre | 29 | -0.2427 | -0.4537 to -0.0221 | 0.1217 |
| shape_variability | post minus pre | 32 | -0.0186 | -0.0251 to -0.0127 | 2.082e-06 |
| shape_variability | retention minus pre | 29 | -0.0089 | -0.0158 to -0.0020 | 0.1216 |


## Does trajectory information add predictive value?

Pre versus post is a chronological phase label, not a measure of automaticity. Each evaluation participant is held out in turn. Scaling, PCA and logistic regression are fitted exclusively on other participants. The model uses no participant ID, condition label, absolute timestamp, block number or trial number as an input. Hyperparameters are fixed: five PCA components, logistic C=1. No tuning was performed on the reported evaluation scores.

| Model | Mean participant balanced accuracy | 95% participant bootstrap interval | Mean AUC |
|---|---:|---|---:|
| Duration only | 0.478 | 0.396 to 0.560 | 0.559 |
| Duration + accuracy | 0.693 | 0.648 to 0.736 | 0.796 |
| Motion features | 0.768 | 0.722 to 0.813 | 0.911 |
| Motion features + PCA | 0.770 | 0.726 to 0.814 | 0.911 |

Adding motion features and PCA changed participant-balanced accuracy by +0.077 compared with duration plus in-channel accuracy (95% participant bootstrap interval +0.024 to +0.130). The interval excludes zero in this internal validation; it does not establish external generalisation or a causal mechanism.

The motion-feature model alone scored 0.768, compared with 0.770 after adding PCA. The point-estimate gain from PCA is therefore negligible; the results support the simpler feature model as a practical baseline, not a claim that dimensionality reduction was essential.

Smoothness changes have positive bootstrap mean intervals, but neither pre/post nor pre/retention LDJ comparison survives the stated Holm correction on Wilcoxon tests. The mean-bootstrap and signed-rank procedures target different summaries; they should not be presented as interchangeable significance evidence.

A subject-level phase-swap diagnostic reverses pre/post labels independently per participant and reruns the full model in 20 fixed repetitions. Its mean balanced accuracy was 0.503, with range 0.386–0.601. This is a diagnostic, not a formal permutation significance claim. Cross-validation folds share training data, so the bootstrap interval is descriptive and does not include all model-fitting uncertainty.

## Smoothing sensitivity

- ldj_short, post minus pre: mean change +0.694, 95% interval [+0.041, +1.346].
- ldj_short, retention minus pre: mean change +0.877, 95% interval [+0.311, +1.506].
- ldj, post minus pre: mean change +0.686, 95% interval [+0.038, +1.342].
- ldj, retention minus pre: mean change +0.842, 95% interval [+0.281, +1.460].
- ldj_long, post minus pre: mean change +0.677, 95% interval [+0.028, +1.347].
- ldj_long, retention minus pre: mean change +0.807, 95% interval [+0.264, +1.381].

## Limits and next evidence

This is one small laboratory task and one dataset. Phase is confounded with elapsed practice, fatigue and session order. Predicting phase does not identify cognitive control, prove automaticity, or establish a causal effect of training schedule. Retention results cover only the participants with consistent recorded identity. The supplied accuracy metric covers the original trial, whereas kinematic features cover the selected segment. Their measurement windows therefore differ.

Endpoint filtering may induce selection bias; no manually annotated onset benchmark is available. Raw camera samples were converted to screen pixels by the original experiment. No physical force or gaze signals are present. External validation on another independently collected motor-learning task and manual boundary annotation would be needed before broader claims. The piano project is not used as motor-learning evidence.

## Authorship and use

Implemented with substantial AI assistance for Hammad Zahid. Before listing it as personal experience, independently review the code, rerun the analysis and explain the methodological choices. The source study and data collection belong to the cited authors. No novel neuroscience discovery, clinical application or publication is claimed. Code and results: https://github.com/Ninjaa-aa/motor-learning-trajectories . No internship application has been submitted.
