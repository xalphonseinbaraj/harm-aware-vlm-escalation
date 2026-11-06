"""Pre-deployment check on the validation and test splits and the external sets (revision R1, Section 5.2 / Table 12),
plus all per-configuration numbers of Tables 7, 10 and 11. Run from the project root:  python revision\\predeployment_check.py
Writes results_rev\\predeployment_check.json."""
import json,glob,numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(1)
def L(f): return {r['image_id']:r for r in map(json.loads,open(f)) if 'vlm_error' not in r}
def M(fs):
    D=[L(f) for f in fs]; ids=sorted(set.intersection(*[set(d) for d in D]))
    g=lambda k:np.stack([[k(d[i]) for i in ids] for d in D])
    bb=g(lambda r:r['bb']['iou']); pr=g(lambda r:r['prop']['iou']); gp=g(lambda r:r['gate_prob']); cf=g(lambda r:r['conf'])
    acc=g(lambda r:(r.get('F') is None) or (r['F']!=r['F']) or r['F']>=0.5); adj=g(lambda r:r['adj']!='no_change')
    return bb,pr,gp,cf,acc,adj
def row(fs):
    bb,pr,gp,cf,acc,adj=M(fs); esc=gp>=0.5; hard=bb<0.7
    gv=np.where(esc&acc,pr,bb); g=np.where(esc,pr,bb)
    rnd=bb+esc.mean(1,keepdims=True)*(pr-bb)
    n=bb.shape[1]; B=[]
    for _ in range(5000):
        ix=rng.integers(0,n,n); nv,al,G=bb[:,ix].mean(),pr[:,ix].mean(),gv[:,ix].mean(); B.append((G-al)/(nv-al) if nv>al else np.nan)
    ha=(gv.mean()-pr.mean())/(bb.mean()-pr.mean()) if bb.mean()>pr.mean() else np.nan
    rej=esc&adj&~acc; harmful=rej&(pr<bb-1e-9)
    au=np.mean([roc_auc_score(hard[s],gp[s]) for s in range(len(bb)) if 0<hard[s].mean()<1])
    auc=np.mean([roc_auc_score(hard[s],-cf[s]) for s in range(len(bb)) if 0<hard[s].mean()<1])
    try: p=wilcoxon(gv.mean(0),bb.mean(0)).pvalue
    except: p=np.nan
    try: pa=wilcoxon(gv.mean(0),pr.mean(0)).pvalue
    except: pa=np.nan
    easy=~hard; eh=np.mean([((pr[s]<bb[s]-1e-9)&easy[s]).sum()/easy[s].sum() for s in range(len(bb))])
    ehmin=min(((pr[s]<bb[s]-1e-9)&easy[s]).sum()/easy[s].sum() for s in range(len(bb))); ehmax=max(((pr[s]<bb[s]-1e-9)&easy[s]).sum()/easy[s].sum() for s in range(len(bb)))
    k=esc.sum(1); o=[]
    for s in range(len(bb)):
        x=bb[s].copy(); t=np.argsort(-(pr[s]-bb[s]))[:k[s]]; x[t]=np.maximum(pr[s][t],bb[s][t]); o.append(x.mean())
    return dict(seeds=len(fs),n=n,hard=hard.mean(),esc=esc.mean(),never=bb.mean(),always=pr.mean(),gate=g.mean(),gateV=gv.mean(),rand=rnd.mean(),
        ha=ha,ha_ci=list(np.nanpercentile(B,[2.5,97.5])),rej=int(rej.sum()),rej_harm=int(harmful.sum()),edits=int((esc&adj).sum()),
        auroc=au,auroc_conf=auc,p_bb=p,p_always=pa,easyhurt=eh,eh_rng=(ehmin,ehmax),ceiling=np.mean(o)-bb.mean(),loss_all=bb.mean()-pr.mean())
out={}
for name,pat in [('K',''),('ClinicDB','CVC-ClinicDB_'),('ColonDB','CVC-ColonDB_'),('ETIS','ETIS-LaribPolypDB_')]:
    for bk in ['segformer','strong']:
        for v in ['medgemma4b_p1','qwen7b_p3','medsam_box']:
            d='results_rev' if name=='K' else 'results_ext'
            fs=sorted(glob.glob(f'{d}/{pat}{bk}_s?__{v}.jsonl'))
            if fs: out[f'{name}|{bk}|{v}']=row(fs)
    if name=='K':
        for bk in ['segformer','strong']:
            for v in ['medgemma4b_p1','qwen7b_p3','medsam_box']:
                fs=sorted(glob.glob(f'results_val/{bk}_s?__{v}.jsonl'))
                if fs: out[f'VAL|{bk}|{v}']=row(fs)
for k,r in out.items():
    print(f"{k:32} s{r['seeds']} hard {r['hard']:.3f} esc {r['esc']:.3f} nv {r['never']:.3f} al {r['always']:.3f} g {r['gate']:.3f} gV {r['gateV']:.3f} rnd {r['rand']:.3f} HA {100*r['ha']:.0f} [{100*r['ha_ci'][0]:.0f}-{100*r['ha_ci'][1]:.0f}] rej {r['rej_harm']}/{r['rej']} of {r['edits']} AU {r['auroc']:.2f}({r['auroc_conf']:.2f}) p_bb {r['p_bb']:.3g} p_al {r['p_always']:.2g} easyhurt {100*r['eh_rng'][0]:.0f}-{100*r['eh_rng'][1]:.0f} ceil {r['ceiling']:+.3f} loss {r['loss_all']:.3f}")
json.dump(out,open("results_rev/predeployment_check.json","w"),indent=1,default=float)
