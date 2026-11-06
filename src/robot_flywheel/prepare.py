"""Download and inspect the original full-state expert archive."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from .data import load_zarr
from .artifacts import write_json

URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

def prepare(directory="data/pusht", expected_sha256=None):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    archive=root/"pusht.zip"
    if not archive.exists():
        temp=root/"pusht.zip.partial"
        urllib.request.urlretrieve(URL,temp)
        temp.replace(archive)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected_sha256 is not None and sha!=expected_sha256:
        raise ValueError("Archive checksum does not match expected source")
    destination=root/"extracted"
    with zipfile.ZipFile(archive) as z:
        # Reject path traversal and symlinks before extracting anything.
        for item in z.infolist():
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (item.external_attr>>16)&0o170000==0o120000:
                raise ValueError("Unsafe archive member")
        z.extractall(destination)
    matches=list(destination.rglob("*.zarr"))
    if len(matches)!=1: raise ValueError(f"Expected one Zarr directory, got {matches}")
    episodes=load_zarr(matches[0])
    states=np.concatenate([e.states for e in episodes]);actions=np.concatenate([e.actions for e in episodes])
    summary=dict(source_url=URL,archive_sha256=sha,zarr_path=str(matches[0].resolve()),episodes=len(episodes),
                 transitions=len(states),state_min=states.min(0).tolist(),state_max=states.max(0).tolist(),
                 action_min=actions.min(0).tolist(),action_max=actions.max(0).tolist(),
                 state_schema=["agent_x","agent_y","object_x","object_y","object_angle"],
                 cadence_hz=10,cadence_provenance="PushT source control_hz; archive has no independent timestamp validation")
    write_json(root/"source.json",summary)
    return episodes,summary
