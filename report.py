"""Build the research report and a participant trajectory explorer from saved evidence."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
LABELS={'duration':'Duration only','duration_accuracy':'Duration + accuracy','kinematics':'Motion features','kinematics_pca':'Motion features + PCA'}
COLORS={'pre':'#b16c39','post':'#137f78','retention':'#675ba3'}


def main():
    result=ROOT/'results';a=result/'analysis';s=json.loads((a/'summary.json').read_text());cfg=json.loads((ROOT/'protocol.json').read_text())
    trials=pd.read_csv(result/'evaluation/trials.csv');shapes=np.load(result/'evaluation/shapes.npz')['shape'];means=pd.read_csv(a/'participant_phases.csv')
    excluded=pd.concat([pd.read_csv(result/x/'exclusions.csv').assign(split=x) for x in ['development','evaluation']])
    counts=excluded.groupby(['split','phase','reason']).size().reset_index(name='count');counts.to_csv(a/'exclusion_counts.csv',index=False)
    table='| Model | Mean participant balanced accuracy | 95% participant bootstrap interval | Mean AUC |\n|---|---:|---|---:|\n'
    for key in LABELS:
        m=s['models'][key];lo,hi=m['balanced_accuracy_bootstrap95'];table+=f"| {LABELS[key]} | {m['balanced_accuracy']:.3f} | {lo:.3f} to {hi:.3f} | {m['auc']:.3f} |\n"
    change='| Measure | Comparison | Participants | Mean change | 95% interval | Holm-adjusted p |\n|---|---|---:|---:|---|---:|\n'
    for row in s['changes']:
        lo,hi=row['bootstrap95'];change+=f"| {row['metric']} | {row['comparison']} | {row['participants']} | {row['mean_change']:.4f} | {lo:.4f} to {hi:.4f} | {row['holm_p']:.4g} |\n"
    delta=s['full_minus_accuracy_baseline'];lo,hi=delta['bootstrap95']
    finding=f"Adding motion features and PCA changed participant-balanced accuracy by {delta['mean']:+.3f} compared with duration plus in-channel accuracy (95% participant bootstrap interval {lo:+.3f} to {hi:+.3f})."
    interpretation='The interval includes zero, so this analysis does not establish a reliable classification improvement.' if lo<=0<=hi else 'The interval excludes zero in this internal validation; it does not establish external generalisation or a causal mechanism.'
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,metric,label in zip(axes[:2],['icf','ldj'],['In-channel fraction (higher is better)','Log dimensionless jerk (higher is smoother)']):
        for p,f in means.groupby('participant'):
            f=f.set_index('phase');phases=['pre','post','retention'];values=[f.loc[x,metric] if x in f.index and not (x=='retention' and p in cfg['retention_identity_mismatch_exclusions']) else np.nan for x in phases]
            ax.plot(range(3),values,color='#96aaaa',alpha=.35,lw=.7)
        for i,ph in enumerate(['pre','post','retention']):
            f=means[means.phase==ph]
            if ph=='retention':f=f[~f.participant.isin(cfg['retention_identity_mismatch_exclusions'])]
            ax.scatter([i],[f[metric].mean()],s=55,color=COLORS[ph],zorder=4)
        ax.set(xticks=range(3),xticklabels=['Pre','Post','Retention'],ylabel=label)
    keys=list(LABELS);values=[s['models'][k]['balanced_accuracy'] for k in keys]
    bounds=np.array([s['models'][k]['balanced_accuracy_bootstrap95'] for k in keys])
    axes[2].bar(range(4),values,color=['#a4b0b8','#6e8b9c','#41a394','#19756d'])
    axes[2].errorbar(range(4),values,yerr=np.array([values-bounds[:,0],bounds[:,1]-values]),fmt='none',color='#293b46',capsize=4)
    axes[2].axhline(.5,color='#a65e4d',ls='--',lw=1)
    axes[2].set(xticks=range(4),xticklabels=['Time','+ Accuracy','Motion','+ PCA'],ylim=(0,1),ylabel='Mean participant balanced accuracy',title='Unseen participant: pre versus post')
    fig.savefig(result/'overview.png',dpi=160);plt.close(fig)
    explorer={}
    for participant,f in trials.groupby('participant'):
        explorer[participant]={}
        for ph,g in f.groupby('phase'):
            curves=shapes[g.index.to_numpy()];explorer[participant][ph]=dict(n=len(g),mean=curves.mean(axis=0).round(5).tolist(),duration=float(g.duration.mean()),accuracy=float(g.icf.mean()),ldj=float(g.ldj.mean()))
    (result/'explorer.json').write_text(json.dumps(explorer)+'\n')
    sensitivity='\n'.join(f"- {r['metric']}, {r['comparison']}: mean change {r['mean_change']:+.3f}, 95% interval [{r['bootstrap95'][0]:+.3f}, {r['bootstrap95'][1]:+.3f}]." for r in s['smoothing_sensitivity'])
    report=f'''# Human motor learning: beyond completion time

## Question and relevance

What do movement smoothness and trajectory shape reveal about practice that duration and task accuracy miss? This project directly addresses CaCTüS Project 9, Learning to Move: Computational Analysis of Human Motor Skills, through behavioural trajectory analysis, numerical signal processing, representation learning and baseline comparisons.

## Data and contribution

Reanalysis of a public wrist-controlled cursor-steering experiment by Vaynee Sungeelee, Antoine Loriette, Olivier Sigaud and Baptiste Caramiaux. The 36 participants performed an arc-tracing task over two days under three practice schedules. This project did not collect the recordings and does not claim to reproduce every result of the authors' paper. It adds a participant-disjoint comparison of phase classifiers, explicit input auditing, a dimensionless smoothness measure, and preprocessing sensitivity checks.

- Dataset version: https://doi.org/10.5281/zenodo.7824331 (CC BY 4.0).
- Primary paper: https://doi.org/10.1038/s41598-024-53253-3 .
- Attribution and licence: https://creativecommons.org/licenses/by/4.0/ .
- Modifications: parsing the original NeDB logs, movement segmentation, resampling, smoothing, feature extraction and analysis. Derived outputs omit raw wall-clock timestamps and generated source usernames.

Four participants ({', '.join(cfg['development_participants'])}) were reserved for development. The remaining {s['evaluation_participants']} participants enter the reported descriptive evaluation; {s['eligible_classification_participants']} satisfy the minimum four valid trials per class for classification. Across development and evaluation, {s['included_trials']} matched-task trials were included and {len(excluded)} were excluded, with every reason recorded. Development participants enter training folds only, never the reported test scores or paired changes.

## Input and identity checks

Participant identity uses the full filename, including condition: P1-CL and P1-EA are different people. The source describes the numeric portion as an ID but repeats numbers across conditions. Three full IDs ({', '.join(cfg['retention_identity_mismatch_exclusions'])}) have differing generated usernames in their day-1/day-2 configurations. Their within-day pre/post comparisons remain valid; their paired retention comparisons are conservatively omitted because identity cannot be independently resolved. Filename-based retention identity is otherwise assumed from the documented archive structure.

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

{change}

## Does trajectory information add predictive value?

Pre versus post is a chronological phase label, not a measure of automaticity. Each evaluation participant is held out in turn. Scaling, PCA and logistic regression are fitted exclusively on other participants. The model uses no participant ID, condition label, absolute timestamp, block number or trial number as an input. Hyperparameters are fixed: five PCA components, logistic C=1. No tuning was performed on the reported evaluation scores.

{table}
{finding} {interpretation}

The motion-feature model alone scored {s['models']['kinematics']['balanced_accuracy']:.3f}, compared with {s['models']['kinematics_pca']['balanced_accuracy']:.3f} after adding PCA. The point-estimate gain from PCA is therefore negligible; the results support the simpler feature model as a practical baseline, not a claim that dimensionality reduction was essential.

Smoothness changes have positive bootstrap mean intervals, but neither pre/post nor pre/retention LDJ comparison survives the stated Holm correction on Wilcoxon tests. The mean-bootstrap and signed-rank procedures target different summaries; they should not be presented as interchangeable significance evidence.

A subject-level phase-swap diagnostic reverses pre/post labels independently per participant and reruns the full model in 20 fixed repetitions. Its mean balanced accuracy was {s['phase_swap_diagnostic']['mean']:.3f}, with range {s['phase_swap_diagnostic']['range'][0]:.3f}–{s['phase_swap_diagnostic']['range'][1]:.3f}. This is a diagnostic, not a formal permutation significance claim. Cross-validation folds share training data, so the bootstrap interval is descriptive and does not include all model-fitting uncertainty.

## Smoothing sensitivity

{sensitivity}

## Limits and next evidence

This is one small laboratory task and one dataset. Phase is confounded with elapsed practice, fatigue and session order. Predicting phase does not identify cognitive control, prove automaticity, or establish a causal effect of training schedule. Retention results cover only the participants with consistent recorded identity. The supplied accuracy metric covers the original trial, whereas kinematic features cover the selected segment. Their measurement windows therefore differ.

Endpoint filtering may induce selection bias; no manually annotated onset benchmark is available. Raw camera samples were converted to screen pixels by the original experiment. No physical force or gaze signals are present. External validation on another independently collected motor-learning task and manual boundary annotation would be needed before broader claims. The piano project is not used as motor-learning evidence.

## Authorship and use

Implemented with substantial AI assistance for Hammad Zahid. Before listing it as personal experience, independently review the code, rerun the analysis and explain the methodological choices. The source study and data collection belong to the cited authors. No novel neuroscience discovery, clinical application or publication is claimed. Code and results: https://github.com/Ninjaa-aa/motor-learning-trajectories . No internship application has been submitted.
'''
    (result/'REPORT.md').write_text(report)
    modelrows=''.join(f"<tr><td>{LABELS[k]}</td><td>{s['models'][k]['balanced_accuracy']:.3f}</td><td>{s['models'][k]['auc']:.3f}</td></tr>" for k in LABELS)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motor learning: beyond completion time</title><style>body{margin:0;background:#f5f7f8;color:#243e48;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:36px 24px}h1{font:42px Georgia,serif}section{background:white;border:1px solid #dae4e6;border-radius:10px;padding:22px;margin:22px 0}img,svg{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#126f75}select{font:inherit;padding:6px}svg{width:100%;background:#fbfcfc}p{max-width:95ch}</style><main><p>CaCTüS Project 9 · Real recorded movement · CPU analysis</p><h1>Motor learning: beyond completion time</h1><p>Do movement smoothness and shape reveal changes that timing and task accuracy miss?</p><p><a href="REPORT.md">Full methods and limitations</a> · <a href="../README.md">Reproduce</a> · <a href="analysis/summary.json">All results</a> · <a href="evaluation/exclusions.csv">Excluded trials</a> · <a href="frozen.json">Frozen protocol</a></p><section><h2>Measured results</h2><p>FINDING</p><img src="overview.png" alt="Paired changes in accuracy and smoothness, and participant-disjoint classification scores"><table><tr><th>Model</th><th>Balanced accuracy</th><th>AUC</th></tr>MODELROWS</table><p>Phase classification is not a measurement of cognitive automaticity. All test participants are excluded from their model's training data.</p></section><section><h2>Inspect a participant's trajectories</h2><label for="participant">Participant code </label><select id="participant"></select><svg id="trajectory" viewBox="0 0 720 430" role="img" aria-label="Mean normalised trajectories before training, after training and at retention"></svg><div id="details" aria-live="polite"></div><p>Mean trajectories are time-normalised and aligned to their own endpoints. Brown: pre-test. Green: post-test. Purple: retention. These plots are descriptive; retention identity mismatches are flagged below.</p></section><section><h2>Evidence and scope</h2><p>36 recorded participants; four development participants and 32 evaluation participants. Original data: Sungeelee, Loriette, Sigaud and Caramiaux, <a href="https://doi.org/10.5281/zenodo.7824331">Zenodo dataset</a>, CC BY 4.0. Source experiment used wrist motion to steer a cursor, not mouse movement or a robotic reaching task.</p><p>Source identities differ across days for P11-EA, P5-CL and P8-RM; paired retention analyses exclude those participants. Signal boundaries are operational estimates and exclude some trials. This is an AI-assisted reanalysis, not newly collected human-experiment data.</p></section></main><script>const data=EXPLORER;const colours={pre:'#b16c39',post:'#137f78',retention:'#675ba3'};const select=document.getElementById('participant');for(const id of Object.keys(data).sort()){let option=document.createElement('option');option.value=id;option.textContent=id;select.appendChild(option)}function draw(){const rows=data[select.value];let paths='';let text='<table><tr><th>Phase</th><th>Trials</th><th>Duration (s)</th><th>Accuracy</th><th>LDJ</th></tr>';for(const ph of ['pre','post','retention']){const r=rows[ph];if(!r)continue;const points=r.mean.map(p=>`${80+p[0]*560},${350+p[1]*560}`).join(' ');paths+=`<polyline fill="none" stroke="${colours[ph]}" stroke-width="3" points="${points}"/>`;text+=`<tr><td>${ph}</td><td>${r.n}</td><td>${r.duration.toFixed(3)}</td><td>${r.accuracy.toFixed(3)}</td><td>${r.ldj.toFixed(3)}</td></tr>`}document.getElementById('trajectory').innerHTML=paths;document.getElementById('details').innerHTML=text+'</table>'+(['P11-EA','P5-CL','P8-RM'].includes(select.value)?'<p>Identity mismatch: retention is shown as an archive record but excluded from paired inference.</p>':'')}select.addEventListener('change',draw);draw();</script></html>'''
    page=page.replace('FINDING',html.escape(finding+' '+interpretation)).replace('MODELROWS',modelrows).replace('EXPLORER',json.dumps(explorer))
    (result/'report.html').write_text(page)
    (ROOT/'CV_EVIDENCE.txt').write_text(f'''Suggested wording after independent review and rerun:

Human motor-learning analysis | Python, SciPy, scikit-learn
Developed an AI-assisted Python pipeline analysing {s['included_trials']:,} recorded wrist-motion trajectories from 36 participants, with signal processing, participant-level statistics and PCA. Evaluated pre/post-training phase classification on {s['eligible_classification_participants']} participants using participant-disjoint folds; motion features achieved {100*s['models']['kinematics']['balanced_accuracy']:.1f}% balanced accuracy versus {100*s['models']['duration_accuracy']['balanced_accuracy']:.1f}% for a timing-and-accuracy baseline.

Do not claim data collection, independent origination, cognitive automaticity, a publication, or external validation. Measured classifier results and uncertainty are in results/REPORT.md.
''')
    print(result/'report.html')


if __name__=='__main__':main()
