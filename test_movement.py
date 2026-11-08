import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movement import segment,describe,phase
from analyse import bootstrap,holm,make_model,predict_folds,scores
from extract import documents

CONFIG=json.loads((Path(__file__).parent/'protocol.json').read_text())


def arc(n=241,duration=2.):
    t=np.linspace(0,duration,n);q=np.pi*t/duration
    return t,np.column_stack([500-400*np.cos(q),500-400*np.sin(q)])


def test_segment_discards_repositioning():
    t,p=arc();prefix=np.array([[900,500],[500,500],[100,500]])
    tt,pp,qc=segment(np.r_[[-.3,-.2,-.1],t],np.vstack([prefix,p]),CONFIG['segmentation'])
    assert tt[0]==0 and 1.5<tt[-1]<2.1 and pp[-1,0]>870 and qc['start_index']>=3


@pytest.mark.parametrize('problem',['nan','clock','missing_end','gap','few'])
def test_segment_rejects_bad_recordings(problem):
    t,p=arc()
    if problem=='nan':p[20,0]=np.nan
    if problem=='clock':t[20]=t[19]
    if problem=='missing_end':p[:,0]/=2
    if problem=='gap':t[100:]+=1
    if problem=='few':t,p=t[:10],p[:10]
    with pytest.raises(ValueError):segment(t,p,CONFIG['segmentation'])


def test_features_have_known_arc_geometry():
    t,p=arc();f,shape=describe(t,p,CONFIG['signal'])
    assert 1.55<f['path_ratio']<1.59 and shape.shape==(64,2)
    np.testing.assert_allclose(shape[0],[0,0],atol=1e-6)
    np.testing.assert_allclose(shape[-1],[1,0],atol=1e-6)
    assert np.isfinite(list(f.values())).all()


def test_dimensionless_jerk_ignores_duration_and_spatial_scale():
    t,p=arc();settings={**CONFIG['signal'],'savgol_window_seconds':.125}
    a,_=describe(t,p,settings)
    b,_=describe(t,p*2+30,settings)
    c,_=describe(t*2,p,{**settings,'resample_hz':60,'savgol_window_seconds':.25})
    assert a['ldj']==pytest.approx(b['ldj'],abs=1e-6)
    assert a['ldj']==pytest.approx(c['ldj'],abs=1e-6)


def test_jitter_reduces_smoothness():
    t,p=arc();f,_=describe(t,p,CONFIG['signal'])
    p[:,1]+=3*np.sin(2*np.pi*12*t)
    g,_=describe(t,p,CONFIG['signal'])
    assert g['ldj']<f['ldj']


@pytest.mark.parametrize('day,block,expected',[(1,0,None),(1,1,'pre'),(1,94,None),(1,95,'post'),(2,0,'retention'),(2,2,None)])
def test_phase_mapping(day,block,expected):assert phase(day,block)==expected


def test_nedb_replays_updates_and_deletions(tmp_path):
    p=tmp_path/'source.db';p.write_text('\n'.join(json.dumps(r) for r in [{'_id':'a','x':1},{'_id':'a','x':2},{'_id':'b'},{'_id':'b','$$deleted':True}]))
    assert list(documents(p))==[{'_id':'a','x':2}]


def test_holm_and_bootstrap():
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert bootstrap([2,2,2],100)==[2,2]


def test_participant_folds_are_disjoint_and_pca_fits_training_only():
    rng=np.random.default_rng(1);n=24
    frame=pd.DataFrame({'participant':np.repeat(['P1-CL','P1-EA','P1-RM'],8),'phase':['pre']*4+['post']*4+['pre']*4+['post']*4+['pre']*4+['post']*4,'condition':'test','block':1,'trial':np.arange(n)})
    for name in ['duration','icf','ldj','path_ratio','radial_rmse']:frame[name]=rng.normal(size=n)
    shape=rng.normal(size=(n,64,2));pred,folds=predict_folds(frame,shape,['P1-CL'],CONFIG['prediction'])
    assert set(pred.participant)=={'P1-CL'} and len(pred)==32
    assert folds[0]['train_participants']==['P1-EA','P1-RM']
    assert len(scores(pred))==4
    x=np.column_stack([frame[['duration','icf','ldj','path_ratio','radial_rmse']],shape.reshape(n,-1)])
    model=make_model('kinematics_pca',CONFIG['prediction']);model.fit(x[8:],(frame.phase=='post').to_numpy()[8:])
    scaler=model.named_steps['representations'].named_transformers_['shape'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_,x[8:,5:].mean(axis=0))
