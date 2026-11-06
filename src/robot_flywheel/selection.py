import numpy as np

def geometry(states, goal=(256.,256.,np.pi/4)):
    s=np.asarray(states); g=np.asarray(goal)
    relative=(s[...,2:4]-g[:2])/512.
    angle=s[...,4:5]-g[2]
    return np.concatenate((relative,np.sin(angle),np.cos(angle),(s[...,:2]-s[...,2:4])/512.,np.linalg.norm(relative,axis=-1,keepdims=True)),axis=-1)

def representatives(episode, count=32):
    return geometry(episode.states[np.linspace(0,len(episode.states)-1,min(count,len(episode.states)),dtype=int)])

def select(pool, acquired, k, strategy, seed=0, failure_features=None):
    candidates={e.id:e for e in pool if e.id not in set(acquired)}
    if not 0<=k<=len(candidates): raise ValueError("Acquisition budget exceeds remaining pool")
    if strategy == "random":
        ids=np.random.default_rng(seed).choice(sorted(candidates),k,replace=False).tolist()
        return ids,[dict(episode_id=i,reason="random") for i in ids]
    if strategy not in ("diversity","targeted"): raise ValueError(strategy)
    reps={e.id:representatives(e) for e in pool}
    fallback = strategy=="targeted" and (failure_features is None or len(failure_features)==0)
    query=np.asarray(failure_features) if strategy=="targeted" and not fallback else np.concatenate(list(reps.values()))
    # Facility-location similarity; diminishing returns discourage redundant selections.
    scores={i:np.exp(-np.min(((query[:,None,:]-x[None,:,:])**2).sum(-1),axis=1)/0.25) for i,x in reps.items()}
    coverage=np.maximum.reduce([scores[i] for i in acquired]) if acquired else np.zeros(len(query))
    chosen=[]; log=[]
    for _ in range(k):
        gains={i:float(np.maximum(coverage,scores[i]).mean()-coverage.mean()) for i in candidates if i not in chosen}
        best=min(gains,key=lambda i:(-gains[i],i))
        chosen.append(best); coverage=np.maximum(coverage,scores[best])
        log.append(dict(episode_id=best,marginal_gain=gains[best],reason="diversity_fallback" if fallback else strategy))
    return chosen,log
