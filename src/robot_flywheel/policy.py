import math
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.a=nn.Conv1d(cin,cout,3,padding=1); self.b=nn.Conv1d(cout,cout,3,padding=1)
        self.norm=nn.GroupNorm(4,cout); self.film=nn.Linear(cond,2*cout)
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,c):
        scale,bias=self.film(c).chunk(2,-1)
        h=self.norm(self.a(x))*(1+scale[:,:,None])+bias[:,:,None]
        return self.b(torch.nn.functional.silu(h))+self.skip(x)

class DiffusionPolicy(nn.Module):
    """Compact conditional temporal U-Net; experimental implementation."""
    def __init__(self, history=2, horizon=16, width=32, steps=100):
        super().__init__()
        if horizon%2 or width%4: raise ValueError("Even horizon and width divisible by four required")
        self.history,self.horizon,self.steps=history,horizon,steps
        self.time=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,64))
        cond=history*6+64
        self.enc=Block(2,width,cond); self.down=nn.Conv1d(width,width,4,stride=2,padding=1)
        self.mid=Block(width,width*2,cond); self.up=nn.ConvTranspose1d(width*2,width,4,stride=2,padding=1)
        self.dec=Block(width*2,width,cond); self.out=nn.Conv1d(width,2,1)
        # Cosine DDPM schedule.
        t=torch.linspace(0,steps,steps+1)/steps
        a=torch.cos((t+.008)/1.008*math.pi/2).square(); a=a/a[0]
        beta=(1-a[1:]/a[:-1]).clamp(max=.999)
        self.register_buffer("beta",beta); self.register_buffer("abar",(1-beta).cumprod(0))
    def forward(self,x,t,obs):
        f=torch.exp(torch.arange(16,device=x.device)*(-math.log(10000)/15))
        phase=t[:,None]*f[None,:]
        c=torch.cat((obs.flatten(1),self.time(torch.cat((phase.sin(),phase.cos()),-1))),-1)
        h=self.enc(x.transpose(1,2),c); z=self.mid(self.down(h),c)
        return self.out(self.dec(torch.cat((h,self.up(z)),1),c)).transpose(1,2)
    def loss(self,obs,actions,mask):
        t=torch.randint(self.steps,(len(obs),),device=obs.device); eps=torch.randn_like(actions)
        a=self.abar[t,None,None]
        noisy=a.sqrt()*actions+(1-a).sqrt()*eps
        error=(self(noisy,t,obs)-eps).square()*mask[:,:,None]
        return error.sum()/(mask.sum()*actions.shape[-1]).clamp_min(1)
    @torch.no_grad()
    def sample(self,obs,generator=None,inference_steps=None):
        if inference_steps is not None:
            if not 1 <= inference_steps <= self.steps: raise ValueError("Invalid inference step count")
            return self.sample_ddim(obs,inference_steps,generator)
        x=torch.randn((len(obs),self.horizon,2),device=obs.device,generator=generator)
        for i in reversed(range(self.steps)):
            t=torch.full((len(obs),),i,device=obs.device,dtype=torch.long)
            b=self.beta[i]; a=1-b; ab=self.abar[i]
            mean=(x-b/(1-ab).sqrt()*self(x,t,obs))/a.sqrt()
            prev=self.abar[i-1] if i else torch.ones_like(ab)
            variance=b*(1-prev)/(1-ab)
            x=mean if i==0 else mean+variance.sqrt()*torch.randn(x.shape,device=x.device,generator=generator)
        return x

    @torch.no_grad()
    def sample_ddim(self,obs,inference_steps,generator=None):
        """Deterministic DDIM trajectory after seeded initial noise; screening only.

        Uses the existing training schedule. Changing sampling changes policy behavior
        and must be recorded in evaluation identity and confirmed before final use.
        """
        x=torch.randn((len(obs),self.horizon,2),device=obs.device,generator=generator)
        schedule=torch.linspace(self.steps-1,0,inference_steps,device=obs.device).round().long()
        for index,timestep in enumerate(schedule):
            t=timestep.expand(len(obs));ab=self.abar[timestep]
            eps=self(x,t,obs);clean=(x-(1-ab).sqrt()*eps)/ab.sqrt()
            prev=self.abar[schedule[index+1]] if index+1<len(schedule) else torch.ones_like(ab)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x
