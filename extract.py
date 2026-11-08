"""Extract only matched-geometry pre/post/retention trials; retain every exclusion."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from movement import phase,segment,describe

ROOT=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def documents(path):
    # NeDB is an append log: keep the latest version of each document and respect tombstones.
    latest={}
    for line in path.open():
        if not line.strip():continue
        row=json.loads(line)
        if '_id' not in row:continue
        if row.get('$$deleted'):latest.pop(row['_id'],None)
        else:latest[row['_id']]=row
    return latest.values()


def run(split):
    config=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results'/split
    if out.exists():raise FileExistsError('Keep prior evidence: '+str(out))
    if split=='evaluation':
        for name,sha in json.loads((ROOT/'results/frozen.json').read_text())['files'].items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen input changed: '+name)
    rows=[];shapes=[];excluded=[];sources=[]
    files=sorted((ROOT/'data/raw').glob('day*/*/*.db'))
    if not files:raise ValueError('No source recordings found.')
    for path in files:
        participant=path.stem;condition=path.stem.split('-')[1]
        development=participant in config['development_participants']
        if development!=(split=='development'):continue
        expected=json.loads((ROOT/'data/source_files.json').read_text())
        if digest(path)!=expected[str(path.relative_to(ROOT))]:raise ValueError('Source checksum mismatch.')
        day=int(path.parts[-3][-1]);groups=defaultdict(list)
        sources.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path)))
        for r in documents(path):
            if 'trial_id' in r and 'block_id' in r:groups[(int(r['block_id']),int(r['trial_id']))].append(r)
        for (block,trial),records in sorted(groups.items()):
            ph=phase(day,block)
            if ph is None:continue
            key=dict(participant=participant,condition=condition,day=day,block=block,trial=trial,phase=ph)
            samples=[r for r in records if all(k in r for k in ['x','y','ts'])]
            try:
                if not samples:raise ValueError('no_samples')
                if any(r['width']!=18 or r['length']!=800 or r.get('mouvement_time')!=1000 for r in samples):raise ValueError('unexpected_geometry_or_tempo')
                ordered=sorted(samples,key=lambda r:r['ts']);unique={}
                for r in ordered:
                    if r['ts'] in unique and (r['x'],r['y'])!=unique[r['ts']]:raise ValueError('conflicting_duplicate_timestamp')
                    unique[r['ts']]=(r['x'],r['y'])
                times=np.array(list(unique));positions=np.array(list(unique.values()))
                icfs=[r['icf'] for r in records if 'x' not in r and 0<=r.get('icf',-1)<=1]
                if len(icfs)!=1:raise ValueError('missing_or_ambiguous_accuracy')
                t,p,qc=segment(times,positions,config['segmentation'])
                features,shape=describe(t,p,config['signal'])
                for label,window in zip(['short','long'],config['signal']['sensitivity_windows_seconds']):
                    alternative,_=describe(t,p,{**config['signal'],'savgol_window_seconds':window})
                    features['ldj_'+label]=alternative['ldj']
                rows.append({**key,'icf':icfs[0],**features,**qc,'duplicate_samples_removed':len(samples)-len(unique)})
                shapes.append(shape)
            except ValueError as error:excluded.append({**key,'reason':str(error)})
        print(split,participant,'day',day,flush=True)
    if not rows:raise ValueError('No valid trials.')
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
    pd.DataFrame(excluded,columns=['participant','condition','day','block','trial','phase','reason']).to_csv(out/'exclusions.csv',index=False)
    np.savez_compressed(out/'shapes.npz',shape=np.asarray(shapes))
    (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(len(rows),'included;',len(excluded),'excluded')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['development','evaluation']);run(p.parse_args().split)
