"""Verify saved metrics, participant isolation, hashes and one reproduced fold."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from threadpoolctl import threadpool_limits

from analyse import predict_folds
from extract import digest

ROOT=Path(__file__).resolve().parent


def main(raw=False):
    r=ROOT/'results';config=json.loads((ROOT/'protocol.json').read_text())
    for name,sha in json.loads((r/'frozen.json').read_text())['files'].items():assert digest(ROOT/name)==sha,name
    if raw:
        for name,sha in json.loads((ROOT/'data/source_files.json').read_text()).items():assert digest(ROOT/name)==sha,name
    frames=[];shapes=[];exclusions=0;fingerprints={}
    for split in ['development','evaluation']:
        f=pd.read_csv(r/split/'trials.csv');x=np.load(r/split/'shapes.npz')['shape']
        assert len(f)==len(x) and x.shape[1:]==(64,2) and np.isfinite(x).all()
        assert not f.duplicated(['participant','day','block','trial']).any()
        assert np.isfinite(f.select_dtypes('number')).all().all() and f.icf.between(0,1).all()
        for p,curve in zip(f.participant,x):
            key=curve.tobytes()
            if key in fingerprints:assert fingerprints[key]==p,'Cross-participant duplicate trajectory'
            fingerprints[key]=p
        frames.append(f);shapes.append(x);exclusions+=len(pd.read_csv(r/split/'exclusions.csv'))
    assert not set(frames[0].participant)&set(frames[1].participant)
    frame=pd.concat(frames,ignore_index=True);shape=np.concatenate(shapes)
    assert len(frame)+exclusions==36*(12+12+24)
    s=json.loads((r/'analysis/summary.json').read_text());assert s['included_trials']==len(frame)
    assert s['evaluation_participants']==frames[1].participant.nunique()==32
    pred=pd.read_csv(r/'analysis/predictions.csv');score=pd.read_csv(r/'analysis/participant_scores.csv')
    assert pred.probability.between(0,1).all()
    for row in score.itertuples():
        f=pred[(pred.participant==row.participant)&(pred.model==row.model)]
        assert np.isclose(balanced_accuracy_score(f.label,f.probability>=.5),row.balanced_accuracy)
        assert np.isclose(roc_auc_score(f.label,f.probability),row.auc)
    for name,f in score.groupby('model'):assert np.isclose(f.balanced_accuracy.mean(),s['models'][name]['balanced_accuracy'])
    folds=json.loads((r/'analysis/folds.json').read_text());assert len(folds)==32
    for fold in folds:assert fold['test_participant'] not in fold['train_participants']
    selected=frame.phase.isin(['pre','post']).to_numpy();subject=folds[0]['test_participant']
    with threadpool_limits(limits=2):repeat,_=predict_folds(frame[selected].reset_index(drop=True),shape[selected],[subject],config['prediction'])
    old=pred[pred.participant==subject].sort_values(['model','block','trial']);repeat=repeat.sort_values(['model','block','trial'])
    np.testing.assert_allclose(old.probability,repeat.probability,atol=1e-10,rtol=1e-10)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and not value.startswith(('http:','https:','#')):assert (r/value).exists(),value
    Links().feed((r/'report.html').read_text())
    receipt=dict(included_trials=len(frame),excluded_trials=exclusions,source_participants=36,evaluation_participants=32,folds_verified=32,model_score_rows=len(score),reproduced_participant=subject,reproduced_models=4,raw_checksums_verified=raw,frozen_hashes_match=True,no_cross_participant_identical_shapes=True,local_report_links_exist=True)
    (r/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',action='store_true');main(p.parse_args().raw)
