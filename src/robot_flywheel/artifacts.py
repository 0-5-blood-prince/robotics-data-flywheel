import hashlib
import json
import os
import tempfile
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def manifest(episodes, parent=None, **metadata):
    ids=[e.id for e in episodes]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate episode IDs")
    body=dict(episode_ids=ids,transitions=sum(len(e.states) for e in episodes),parent=parent,**metadata)
    return dict(body,sha256=digest(body))
