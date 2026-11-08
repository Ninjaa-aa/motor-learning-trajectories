"""Validate development-only folds before freezing the evaluation protocol."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyse import predict_folds

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    config=json.loads((ROOT/'protocol.json').read_text());r=ROOT/'results/development'
    frame=pd.read_csv(r/'trials.csv');shape=np.load(r/'shapes.npz')['shape']
    assert set(frame.participant)==set(config['development_participants'])
    selected=frame.phase.isin(['pre','post']).to_numpy()
    with threadpool_limits(limits=2):
        pred,folds=predict_folds(frame[selected].reset_index(drop=True),shape[selected],config['development_participants'],config['prediction'])
    assert pred.probability.between(0,1).all()
    receipt=dict(included_trials=len(frame),folds=len(folds),models=4,finite_predictions=True)
    (r/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)
