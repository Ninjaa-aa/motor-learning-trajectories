# Human motor learning: beyond completion time

A CPU research project for CaCTüS 2027 **Project 9: Learning to Move: Computational Analysis of Human Motor Skills**.

**Question:** What do trajectory smoothness and shape reveal about practice that duration and task accuracy miss?

## Results and artifacts

Open [the offline report](results/report.html) for the measured results and an interactive participant trajectory explorer. Read [the research report](results/REPORT.md) for the protocol, exclusions, uncertainty and limitations. Every prediction and participant-level score is in `results/analysis/`.

The source experiment contains 36 participants performing a wrist-controlled cursor task across two days. Four participants were used for development. Evaluation uses 32 other participants with leave-one-participant-out fitting. Each test person's records are excluded from their model's training data, including scaling and PCA.

| Model | Mean participant balanced accuracy |
|---|---:|
| Duration only | 0.478 |
| Duration + in-channel accuracy | 0.693 |
| Motion features | 0.768 |
| Motion features + five PCA components | 0.770 |

The full model improves on duration plus accuracy by 7.7 percentage points; the descriptive participant bootstrap interval is +2.4 to +13.0 points. PCA adds little beyond simple motion features. A subject-level label-swap diagnostic averages 0.503 over 20 repetitions.

These are internal cross-validation results for **pre-test versus post-test phase**, not an automaticity detector, skill diagnosis or externally validated prediction system. The small sample and shared training data across folds limit the uncertainty estimates.

## Why this fits the internship and Hammad's background

The internship calls for analysis of sensorimotor trajectories, comparison with conventional performance measures, and assessment of what representations reveal about learning. This project does those tasks on real recorded movements with known practice phases.

The work extends existing Python research-software experience, signal processing from the YSP sonification work, statistical evaluation from the LLM-code study, and representation analysis from continual video classification. Those are transferable skills. This project does not turn those prior experiences into a claim of collecting human movement data or conducting human experiments.

## Reproduce

Python 3.13 was used. No GPU, pretrained model or additional paid service is needed. The source archive is 162 MB and expands to roughly a gigabyte; allow about 2 GB of disk space. The downloaded archive and raw files are excluded from the deliverable ZIP; the public downloader retrieves the exact version with published checksums.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python download.py
python extract.py development
python validate_development.py
python freeze.py
python extract.py evaluation
python analyse.py
python report.py
python verify.py --raw
```

The package already includes the measured outputs. For a clean rerun, copy the project into another directory and move that copy's `results/` directory to a backup before running the sequence above. The runners refuse to overwrite existing extraction/analysis results or a frozen protocol. Keep the original frozen results as the comparison record.

To check supplied outputs without downloading raw data:

```bash
pytest -q
python verify.py
```

The audit recomputes metrics, checks participant separation, inspects saved arrays and links, and refits all four models for one evaluation participant. Use `--raw` after downloading to check every raw-file checksum as well.

## Method

- Read original NeDB JSON logs, replay document updates and deletions, and remove identical duplicate sample timestamps.
- Match task diameter, channel width and instructed movement time across pre-test, post-test and retention. Do not compare training tasks with different difficulty as if they were the same task.
- Use operational left-to-right endpoint boundaries to remove waiting and repositioning. Reject missing endpoints, insufficient samples, conflicting timestamps and large sample gaps; retain every exclusion.
- Resample the camera clock to 120 Hz. Use cubic Savitzky–Golay smoothing, with the primary 125 ms window and fixed 83/167 ms sensitivity checks.
- Measure duration, source in-channel fraction, log dimensionless jerk, path-length/displacement ratio and radial error. Represent aligned trajectories with 64 time-normalised points and five PCA components.
- Fit fixed logistic-regression baselines inside participant-disjoint folds. No participant identifiers, condition labels, absolute timestamps, trial numbers or phase metadata enter the predictors.
- Estimate changes using paired participant means, bootstrap intervals and Holm-corrected Wilcoxon tests. Keep immediate post-training performance separate from next-day retention.

`results/frozen.json` records implementation, configuration and source-manifest hashes before evaluation. `results/development_initial/` preserves the initial development extraction before adding smoothing sensitivity. No evaluation-driven tuning was performed.

## Data quality and limits

Full IDs include training condition: P1-CL and P1-EA are different participants. Three records have inconsistent generated usernames across days; paired retention analyses exclude P11-EA, P5-CL and P8-RM. These participants remain eligible for within-session pre/post comparisons. The other cross-day matches follow the source archive's participant labels.

Endpoint-based exclusions may favour better movements. Raw coordinates are screen pixels derived from wrist motion, not calibrated 3D positions. Accuracy is supplied by the original software over its trial window; kinematic features use the extracted movement segment. The jerk definition and segmentation differ from the original paper, so this is not an exact replication of its numerical results. There are no gaze or force measurements in this dataset. Fatigue and session order remain potential confounds. No causal claim about automaticity or training schedules is made.

The simpler motion-feature model performs almost as well as the PCA model. This negative ablation result is retained. Mean smoothness changes have positive bootstrap intervals but do not pass the stated multiple-comparison-corrected signed-rank tests; the report distinguishes those summaries.

## Attribution

Dataset: Vaynee Sungeelee, Antoine Loriette, Olivier Sigaud and Baptiste Caramiaux (2023), *Interactive Curriculum Learning Decreases and Homogenizes Motor Variability*. Version 1, https://doi.org/10.5281/zenodo.7824331 . Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source paper: Sungeelee et al. (2024), *Interactive curriculum learning increases and homogenizes motor smoothness*, Scientific Reports 14, 2843. https://doi.org/10.1038/s41598-024-53253-3 . The source authors designed and conducted the human experiment. This project modifies and reanalyses their data; it does not claim their data collection or scientific findings as original work.

The initially investigated exoskeleton reaching dataset (Dryad 10.5061/dryad.69p8cz9k7) was not used because its download endpoints returned access errors. No robotic-reaching claim is made for the dataset actually analysed here.

## Authorship and CV use

Substantial AI coding and research assistance was used at Hammad Zahid's request. Independently review, rerun and understand the implementation before using the suggested wording in `CV_EVIDENCE.txt`. Describe the contribution as analysis software and secondary analysis of public recordings. Do not claim independent origination, a publication, clinical validation or human-participant data collection.

Public repository: https://github.com/Ninjaa-aa/motor-learning-trajectories . No internship application has been submitted.
