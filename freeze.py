"""Freeze the analysis before examining evaluation outcomes."""
import json
from datetime import datetime,timezone
from pathlib import Path
from extract import digest

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    output=ROOT/'results/frozen.json'
    if output.exists():raise FileExistsError('A frozen protocol already exists.')
    if not (ROOT/'results/development/validation.json').exists():raise ValueError('Complete development checks first.')
    files=['movement.py','extract.py','analyse.py','protocol.json','requirements.txt','download.py','test_movement.py','data/manifest.json','data/source_files.json']
    output.write_text(json.dumps(dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),files={p:digest(ROOT/p) for p in files}),indent=2)+'\n')
    print(output)
