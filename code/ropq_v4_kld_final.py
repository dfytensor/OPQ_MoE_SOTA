"""
ROQ v4 KLD Benchmark - FINAL MINIMAL VERSION
=============================================
Goal: Get reliable KLD numbers for all 7 methods quickly.
Model: 64 dim, 2 layers, 16 seq (tiny but captures distribution shifts)
Focus: relative ranking of methods + bit-width sweep.
"""
import numpy as np
import time

np.random.seed(0)

# ============== OPQ ==============
class Q:
    def __init__(self, b=4, a=0.45, stoch=False):
        self.b, self.a, self.s_flag = b, a, stoch
        self.L = 1<<(b-1); self.sc = None
    def f(self, W):
        mx = max(np.max(np.abs(W)), 1e-30)
        self.sc = (self.L-1)/(mx**self.a)
    def q(self, W):
        z = np.abs(W)**self.a * self.sc
        if self.s_flag:
            fl=np.floor(z); r=np.random.random(z.shape)
            qm=(fl+(r<z-fl)).clip(0,self.L-1)
        else:
            qm=np.clip(np.round(z),0,self.L-1)
        sb=1<<(self.b-1)
        return (qm.astype(np.int64)|((W<0).astype(np.int64)<<sb)).astype(np.uint8)
    def dq(self, q):
        sb=1<<(self.b-1)
        sg=np.where((q&sb)!=0,-1.,1.)
        qm=(q&(sb-1)).astype(float)
        return sg*(np.clip(qm/self.sc,1e-30,None)**(1/self.a))
    def quant(self, W): self.f(W); return self.q(W)
    def dequant(self, q): return self.dq(q)

def ba(ch, bits):
    best_m, best = float('inf'), 0.45
    for a in [0.25,0.30,0.35,0.40,0.45,0.50,0.55]:
        q=Q(bits,a); q.f(ch)
        m=np.mean((ch-q.dq(q.q(ch)))**2)
        if m<best_m: best_m,best=m,a
    return best

# ============== Model ==============
V,D,NH,FF,L,SL = 1000,64,4,4,2,16
W = {}
W['emb'] = np.random.randn(V,D)*0.02
for i in range(L):
    sc = 1+0.3*i
    for k in ['q','k','v']: W[f'{i}{k}']=np.random.randn(D,D)*(0.02/np.sqrt(i+1))*sc
    W[f'{i}o']=np.random.randn(D,D)*0.02
    W[f'{i}g']=np.random.randn(D*FF,D)*(0.04/np.sqrt(i+1))*sc
    W[f'{i}u']=np.random.randn(D*FF,D)*(0.04/np.sqrt(i+1))*sc
    W[f'{i}d']=np.random.randn(D,D*FF)*0.02
    W[f'{i}lg']=np.ones(D); W[f'{i}lb']=np.zeros(D)
    W[f'{i}2g']=np.ones(D); W[f'{i}2b']=np.zeros(D)
W['lm']=np.random.randn(V,D)*0.02

NP = sum(w.size for w in W.values())
print(f"Params: {NP:,}, FP16={NP*2/8/1e6:.3f}MB")

# ============== Forward ==============
def sm(x):
    x=x-x.max(-1,keepdims=True)
    e=np.exp(x); return e/e.sum(-1,keepdims=True)

def forward(qw):
    h=qw['emb'][tokens]
    states=[]
    for i in range(L):
        g=W[f'{i}lg']; b=W[f'{i}lb']
        mu=h.mean(-1,keepdims=True); vr=h.var(-1,keepdims=True)+1e-5
        x=g*(h-mu)/np.sqrt(vr)+b
        Q=x@qw[f'{i}q'].T; K=x@qw[f'{i}k'].T; Vv=x@qw[f'{i}v'].T
        d=Q.shape[-1]
        sc=Q@K.T/np.sqrt(d); sc=sc-sc.max(-1,keepdims=True)
        a=sm(sc); ctx=a@Vv
        h=h+ctx@qw[f'{i}o'].T
        g2=W[f'{i}2g']; b2=W[f'{i}2b']
        mu2=h.mean(-1,keepdims=True); vr2=h.var(-1,keepdims=True)+1e-5
        x2=g2*(h-mu2)/np.sqrt(vr2)+b2
        go=x2@qw[f'{i}g'].T; uo=x2@qw[f'{i}u'].T
        # GELU
        go=0.5*go*(1+np.tanh(np.sqrt(2/np.pi)*(go+0.044715*go**3)))
        h=h+(go*uo)@qw[f'{i}d'].T
        states.append(h.copy())
    return h@qw['lm'].T, states

# ============== KLD ==============
def kld(pq, qq):
    p=sm(pq); q=sm(qq)
    p=np.clip(p,1e-12,1); q=np.clip(q,1e-12,1)
    return np.mean(np.sum(p*np.log(p/q),-1))

# ============== Methods ==============
def m_global(b=4,a=0.45):
    qw={}
    for n,w in W.items():
        q=Q(b,a); q.f(w); qw[n]=q.dq(q.q(w))
    return qw

def m_perch(b=4):
    qw={}
    for n,w in W.items():
        Wh=np.zeros_like(w)
        for i in range(w.shape[0]):
            a=ba(w[i],b)
            q=Q(b,a); q.f(w[i])
            Wh[i]=q.dq(q.q(w[i]))
        qw[n]=Wh
    return qw

def m_dtpc(b=4):
    taus=[0.01,0.015,0.02,0.03,0.05]
    qw={}
    for n,w in W.items():
        best_m,best=None,None
        for tau in taus:
            Wh=np.zeros_like(w)
            lm=np.abs(w)>=tau; sm=np.abs(w)<tau
            for mask,hi,lo in [(lm,0.55,0.40),(sm,0.40,0.25)]:
                for i in range(w.shape[0]):
                    m=mask[i]
                    if m.sum()==0:continue
                    ch=w[i,m]; a=ba(ch,b)
                    q=Q(b,a); q.f(ch)
                    Wh[i,m]=q.dq(q.q(ch))
            mt=np.mean((w-Wh)**2)
            if best_m is None or mt<best_m: best_m,best=mt,Wh.copy()
        qw[n]=best
    return qw

def m_stoch(b=4,N=4):
    taus=[0.01,0.015,0.02,0.03,0.05]
    qw={}
    for n,w in W.items():
        all_W=[]
        for seed in range(N):
            np.random.seed(seed*7+42)
            best_m,best=None,None
            for tau in taus:
                Wh=np.zeros_like(w)
                lm=np.abs(w)>=tau; sm=np.abs(w)<tau
                for mask,hi,lo in [(lm,0.55,0.40),(sm,0.40,0.25)]:
                    for i in range(w.shape[0]):
                        m=mask[i]
                        if m.sum()==0:continue
                        a=ba(w[i,m],b)
                        q=Q(b,a); q.f(w[i,m])
                        # Stochastic rounding manually
                        z=np.abs(w[i,m])**a*q.sc
                        fl=np.floor(z); fr=z-fl
                        r=np.random.random(z.shape)
                        qm=(fl+(r<fr)).clip(0,q.L-1)
                        # dequant stochastic
                        sg=np.where(w[i,m]<0,-1.,1.)
                        x=np.clip(qm/q.sc,1e-30,None)
                        Wh[i,m]=sg*(x**(1/a))
                mt=np.mean((w-Wh)**2)
                if best_m is None or mt<best_m: best_m,best=mt,Wh.copy()
            all_W.append(best)
        np.random.seed(0)
        qw[n]=np.mean(all_W,axis=0)
    return qw

def m_resid(bm=4,br=2):
    qw={}
    for n,w in W.items():
        W1=np.zeros_like(w)
        for i in range(w.shape[0]):
            a=ba(w[i],bm); q=Q(bm,a); q.f(w[i])
            W1[i]=q.dq(q.q(w[i]))
        R=w-W1; q2=Q(br,0.30); q2.f(R)
        qw[n]=W1+q2.dq(q2.q(R))
    return qw

def m_mixed():
    qw={}
    for n,w in W.items():
        Wh=np.zeros_like(w)
        nr=np.linalg.norm(w,1)
        order=np.argsort(nr)[::-1]
        asn=np.ones(w.shape[0],int)*4
        t=max(w.shape[0]//3,1)
        for k,idx in enumerate(order):
            if k<t: asn[idx]=5
            elif k>=2*t: asn[idx]=3
        for i in range(w.shape[0]):
            a=ba(w[i],asn[i]); q=Q(asn[i],a); q.f(w[i])
            Wh[i]=q.dq(q.q(w[i]))
        qw[n]=Wh
    return qw

def m_ln():
    qw={}
    for n,w in W.items():
        if w.ndim==1:
            # 1D weight: just quantize directly
            a=ba(w.reshape(1,-1),4)
            q=Q(4,a); q.f(w)
            qw[n]=q.dq(q.q(w))
            continue
        rv=np.var(w,1,keepdims=True)
        tg=np.sqrt(max(rv.mean(),1e-8))
        sc=tg/np.sqrt(np.maximum(rv,1e-8))
        ws=w*sc
        Wh=np.zeros_like(w)
        for i in range(w.shape[0]):
            a=ba(ws[i],4); q=Q(4,a); q.f(ws[i])
            Wh[i]=q.dq(q.q(ws[i]))
        qw[n]=Wh/np.maximum(sc,1e-30)
    return qw

# ============== Run ==============
print("Generating prompts...")
tokens = np.random.randint(0,500,SL)
N_TRIALS=10
prompts=[np.random.randint(0,500,SL) for _ in range(N_TRIALS)]

print("Baseline...")
t0=time.time()
base=[]
for tok in prompts:
    logits,st=forward(W)
    base.append((logits,st))
print(f"  {time.time()-t0:.1f}s")

methods=[
    ('A) Global OPQ',   lambda:m_global(4,0.45)),
    ('B) Per-ch alpha', m_perch),
    ('C) DT+PC',        lambda:m_dtpc(4)),
    ('D) DT+PC+Stoch',  lambda:m_stoch(4,4)),
    ('E) Residual',      m_resid),
    ('F) Mixed',         m_mixed),
    ('G) LN-Aware',      m_ln),
]

print(f"\n{'='*72}")
print(f"{'Method':>20} | {'KLD':>10} | {'Storage':>8} | {'vsFP16':>6} | {'Status':>8}")
print(f"{'='*72}")

results={}
for name,mfn in methods:
    t0=time.time()
    qw=mfn()
    # Effective bits
    if 'Mixed' in name: eff=4.0
    elif 'Residual' in name: eff=4.5
    else: eff=4.0
    st_mb=NP*eff/8/1e6
    klds=[]
    for idx,tok in enumerate(prompts):
        lq,_=forward(qw)
        klds.append(kld(base[idx][0],lq))
    avg=np.mean(klds); std=np.std(klds)
    ratio=eff/2.0
    st="USABLE" if avg<0.6 else "DEGRADED"
    results[name]={'kld':avg,'std':std,'eff':eff}
    print(f"  {name:>18} | {avg:>10.5f} | {st_mb:>6.2f}MB | {ratio:>5.2f}x | {st:>8}  [{time.time()-t0:.0f}s]")

# Bit sweep
print(f"\n{'-'*55}")
print("Bit sweep (Per-ch alpha):")
print(f"{'Bits':>6} | {'KLD':>10} | {'Storage MB':>10} | {'Status':>8}")
for b in [2,3,4,5,6,8]:
    qw={}
    for n,w in W.items():
        Wh=np.zeros_like(w)
        for i in range(w.shape[0]):
            a=ba(w[i],b); q=Q(b,a); q.f(w[i])
            Wh[i]=q.dq(q.q(w[i]))
        qw[n]=Wh
    klds=[]
    for idx,tok in enumerate(prompts[:5]):
        lq,_=forward(qw)
        klds.append(kld(base[idx][0],lq))
    avg=np.mean(klds)
    mb=NP*b/8/1e6
    st="USABLE" if avg<0.6 else "DEGRADED"
    print(f"  {b:>4}bit | {avg:>10.5f} | {mb:>10.2f} | {st:>8}")

# Per-layer for winner
print(f"\n{'-'*40}")
print("Per-layer KLD (DT+PC+Stoch):")
qw=m_stoch(4,4)
lklds_p=[]
for idx,tok in enumerate(prompts[:5]):
    _,stq=forward(qw)
    lklds_p.append([kld(base[idx][1][i],stq[i]) for i in range(L)])
for i in range(L):
    vals=[v[i] for v in lklds_p]
    print(f"  Layer {i}: KLD={np.mean(vals):.5f}")

np.savez('/data/workspace/v4_kld_final.npz', results=results)
print("\nDone!")
