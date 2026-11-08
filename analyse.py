"""Participant-level changes and participant-disjoint phase classification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
FEATURES=['duration','icf','ldj','path_ratio','radial_rmse']
MODELS=['duration','duration_accuracy','kinematics','kinematics_pca']


def bootstrap(values,repetitions=5000,seed=2027):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid bootstrap values.')
    rng=np.random.default_rng(seed)
    return np.percentile(rng.choice(values,(repetitions,len(values)),replace=True).mean(axis=1),[2.5,97.5]).tolist()


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    out=np.empty_like(p);out[order]=np.minimum(1,adjusted);return out


def make_model(name,config):
    count={'duration':1,'duration_accuracy':2,'kinematics':5,'kinematics_pca':5}[name]
    transformers=[('features',StandardScaler(),list(range(count)))]
    if name=='kinematics_pca':
        transformers.append(('shape',Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=config['pca_components'],svd_solver='full'))]),list(range(5,133))))
    return Pipeline([('representations',ColumnTransformer(transformers)),('classifier',LogisticRegression(C=config['logistic_C'],max_iter=1000,class_weight='balanced',random_state=config['seed']))])


def predict_folds(frame,shapes,participants,config,models=MODELS,permutation_seed=None):
    x=np.column_stack([frame[FEATURES].to_numpy(),shapes.reshape(len(shapes),-1)])
    y=(frame.phase=='post').to_numpy().astype(int);groups=frame.participant.to_numpy()
    if permutation_seed is not None:
        rng=np.random.default_rng(permutation_seed)
        for subject in sorted(set(groups)):
            if rng.random()<.5:y[groups==subject]=1-y[groups==subject]
    rows=[];folds=[]
    for subject in participants:
        test=groups==subject;train=~test
        assert not set(groups[train])&set(groups[test])
        if len(np.unique(y[test]))!=2 or len(np.unique(y[train]))!=2:raise ValueError('Both phases required in every fold.')
        folds.append(dict(test_participant=subject,train_participants=sorted(set(groups[train])),train_trials=int(train.sum()),test_trials=int(test.sum())))
        for name in models:
            model=make_model(name,config);model.fit(x[train],y[train]);prob=model.predict_proba(x[test])[:,1]
            for i,p in zip(np.flatnonzero(test),prob):
                rows.append(dict(participant=subject,condition=frame.iloc[i].condition,phase=frame.iloc[i].phase,block=int(frame.iloc[i].block),trial=int(frame.iloc[i].trial),label=int(y[i]),model=name,probability=float(p)))
    return pd.DataFrame(rows),folds


def scores(predictions):
    rows=[]
    for (participant,model),f in predictions.groupby(['participant','model']):
        rows.append(dict(participant=participant,model=model,balanced_accuracy=balanced_accuracy_score(f.label,f.probability>=.5),auc=roc_auc_score(f.label,f.probability)))
    return pd.DataFrame(rows)


def main():
    config=json.loads((ROOT/'protocol.json').read_text());result=ROOT/'results'
    for name,sha in json.loads((result/'frozen.json').read_text())['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen file changed: '+name)
    out=result/'analysis'
    if out.exists():raise FileExistsError('Preserve the previous analysis.')
    frames=[];shapes=[]
    for split in ['development','evaluation']:
        frames.append(pd.read_csv(result/split/'trials.csv'));shapes.append(np.load(result/split/'shapes.npz')['shape'])
    data=pd.concat(frames,ignore_index=True);shape=np.concatenate(shapes)
    evaluation=data[~data.participant.isin(config['development_participants'])].copy()
    counts=evaluation.groupby(['participant','phase']).size().unstack(fill_value=0)
    eligible=counts.index[(counts.get('pre',0)>=4)&(counts.get('post',0)>=4)].tolist()
    out.mkdir();counts.to_csv(out/'phase_counts.csv')
    phase_means=evaluation.groupby(['participant','condition','phase'])[FEATURES].mean().reset_index()
    variability=[]
    for (p,ph),indices in evaluation.groupby(['participant','phase']).groups.items():
        selected=shape[np.asarray(list(indices))]
        variability.append(dict(participant=p,phase=ph,shape_variability=float(np.sqrt(np.mean((selected-selected.mean(axis=0))**2)))))
    phase_means=phase_means.merge(pd.DataFrame(variability),on=['participant','phase'])
    phase_means.to_csv(out/'participant_phases.csv',index=False)
    changes=[]
    for metric in FEATURES+['shape_variability']:
        table=phase_means.pivot(index='participant',columns='phase',values=metric)
        for ph in ['post','retention']:
            pairs=table[['pre',ph]].dropna()
            if ph=='retention':pairs=pairs.drop(index=config['retention_identity_mismatch_exclusions'],errors='ignore')
            diff=(pairs[ph]-pairs.pre).to_numpy()
            pvalue=float(wilcoxon(diff).pvalue) if np.any(diff) else 1.
            changes.append(dict(metric=metric,comparison=ph+' minus pre',participants=len(diff),pre_mean=float(pairs.pre.mean()),later_mean=float(pairs[ph].mean()),mean_change=float(diff.mean()),bootstrap95=bootstrap(diff,config['bootstrap_repetitions']),wilcoxon_p=pvalue))
    for row,p in zip(changes,holm([x['wilcoxon_p'] for x in changes])):row['holm_p']=float(p)
    selected=data.phase.isin(['pre','post']).to_numpy();frame=data.loc[selected].reset_index(drop=True);features=shape[selected]
    with threadpool_limits(limits=2):
        predictions,folds=predict_folds(frame,features,eligible,config['prediction'])
    predictions.to_csv(out/'predictions.csv',index=False);performance=scores(predictions);performance.to_csv(out/'participant_scores.csv',index=False)
    summary={}
    for name,f in performance.groupby('model'):
        summary[name]=dict(participants=len(f),balanced_accuracy=float(f.balanced_accuracy.mean()),balanced_accuracy_bootstrap95=bootstrap(f.balanced_accuracy),auc=float(f.auc.mean()),auc_bootstrap95=bootstrap(f.auc))
    paired=performance.pivot(index='participant',columns='model',values='balanced_accuracy')
    difference=(paired.kinematics_pca-paired.duration_accuracy).to_numpy()
    diagnostic=[]
    with threadpool_limits(limits=2):
        for seed in range(2027,2027+config['prediction']['phase_swap_repetitions']):
            perm,_=predict_folds(frame,features,eligible,config['prediction'],['kinematics_pca'],permutation_seed=seed)
            diagnostic.append(float(scores(perm).balanced_accuracy.mean()))
    sensitivity=[]
    for metric in ['ldj_short','ldj','ldj_long']:
        means=evaluation.groupby(['participant','phase'])[metric].mean().unstack()
        for ph in ['post','retention']:
            pairs=means[['pre',ph]].dropna()
            if ph=='retention':pairs=pairs.drop(index=config['retention_identity_mismatch_exclusions'],errors='ignore')
            change=(pairs[ph]-pairs.pre).to_numpy()
            sensitivity.append(dict(metric=metric,comparison=ph+' minus pre',participants=len(change),mean_change=float(change.mean()),bootstrap95=bootstrap(change)))
    report=dict(smoothing_sensitivity=sensitivity,dataset_participants=int(data.participant.nunique()),evaluation_participants=int(evaluation.participant.nunique()),eligible_classification_participants=len(eligible),included_trials=len(data),evaluation_trials=len(evaluation),changes=changes,models=summary,full_minus_accuracy_baseline=dict(mean=float(difference.mean()),bootstrap95=bootstrap(difference)),phase_swap_diagnostic=dict(repetitions=20,mean=float(np.mean(diagnostic)),range=[min(diagnostic),max(diagnostic)],scores=diagnostic))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['eligible_classification_participants','models','full_minus_accuracy_baseline','phase_swap_diagnostic']},indent=2))


if __name__=='__main__':main()
