"""Retrieve the versioned, CC-BY 4.0 source and verify its published checksum."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
URL='https://zenodo.org/api/records/7824331/files/Interactive%20CL.zip/content'
MD5='bdfd37d21aef2fc756a52a2269fe1d97'
SIZE=162473113


def main():
    data=ROOT/'data';data.mkdir(exist_ok=True)
    archive=data/'InteractiveCL.zip'
    if not archive.exists():
        partial=archive.with_suffix('.zip.part')
        with urllib.request.urlopen(URL,timeout=90) as response,partial.open('wb') as output:
            while block:=response.read(1024*1024):output.write(block)
        if partial.stat().st_size!=SIZE or hashlib.file_digest(partial.open('rb'),'md5').hexdigest()!=MD5:
            raise ValueError('Source download checksum/size mismatch; partial file retained.')
        partial.rename(archive)
    assert archive.stat().st_size==SIZE
    assert hashlib.file_digest(archive.open('rb'),'md5').hexdigest()==MD5
    sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    manifest=dict(doi='10.5281/zenodo.7824331',url=URL,bytes=SIZE,md5=MD5,sha256=sha,license='CC-BY-4.0',authors=['Vaynee Sungeelee','Antoine Loriette','Olivier Sigaud','Baptiste Caramiaux'])
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=data/'raw';target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.filename.startswith('__MACOSX/') or not info.filename.endswith(('.db','.json')):continue
            dest=(target/info.filename).resolve()
            if not dest.is_relative_to(target.resolve()):raise ValueError('Unsafe archive path.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            content=z.read(info)
            if dest.exists() and dest.read_bytes()!=content:raise ValueError('Existing source file differs.')
            dest.write_bytes(content)
    sources={}
    for p in sorted(target.glob('day*/*/*')):
        if p.is_file():
            with p.open('rb') as f:sources[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    (data/'source_files.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Verified and extracted',SIZE,'bytes')


if __name__=='__main__':main()
