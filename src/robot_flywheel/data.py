from dataclasses import dataclass
import numpy as np

@dataclass
class Episode:
    id: int
    states: np.ndarray
    actions: np.ndarray
    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.states.ndim != 2 or self.states.shape[1] != 5:
            raise ValueError("Need full [agent_x, agent_y, object_x, object_y, angle] state")
        if self.actions.shape != (len(self.states), 2) or not len(self.states):
            raise ValueError("Aligned nonempty states/actions required")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Nonfinite episode")

def encode(states):
    s = np.asarray(states)
    return np.concatenate((s[..., :4], np.sin(s[..., 4:5]), np.cos(s[..., 4:5])), axis=-1).astype(np.float32)

class Normalizer:
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)
    @classmethod
    def fit(cls, arrays):
        x = np.concatenate(arrays)
        return cls(x.mean(0), np.maximum(x.std(0), 1e-3))
    def transform(self, x): return ((x-self.mean)/self.scale).astype(np.float32)
    def inverse(self, x): return x*self.scale+self.mean
    def to_dict(self): return dict(mean=self.mean.tolist(), scale=self.scale.tolist())

def window(episode, t, history=2, horizon=16):
    if not 0 <= t < len(episode.states): raise IndexError(t)
    oi = np.maximum(np.arange(t-history+1,t+1),0)
    ai = np.arange(t,t+horizon)
    mask = ai < len(episode.actions)
    return encode(episode.states[oi]), episode.actions[np.minimum(ai,len(episode.actions)-1)], mask

def load_zarr(path):
    import zarr
    root = zarr.open(str(path), mode="r")
    states, actions = root["data/state"][:], root["data/action"][:]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=int)
    if len(ends)==0 or ends[-1]!=len(states) or len(actions)!=len(states) or np.any(np.diff(np.r_[0,ends])<=0):
        raise ValueError("Invalid episode boundaries")
    return [Episode(i,states[a:b],actions[a:b]) for i,(a,b) in enumerate(zip(np.r_[0,ends[:-1]],ends))]
