"""Learned acceptance rule for MedSAM (and the VLMs), revision R1, Section 4.9 / Table 11.
Fits a logistic regression on the VALIDATION split only (seeds pooled) to predict whether the escalated output raises IoU,
from seven inference-time inputs: faithfulness score F, log relative size of the change, the four gate features, gate probability.
Applies it unchanged, once, to the Kvasir-SEG test split and the three external datasets. Run from the project root:
    python revision\\learned_acceptance.py
Writes results_rev\\learned_acceptance.json."""
import json,glob,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
rng=np.random.default_rng(0)
def L(f): return {r['image_id']:r for r in map(json.loads,open(f)) if 'vlm_error' not in r}
def X(r):
    c,a,e,b=r['feats']; F=r.get('F'); F=0.0 if (F is None or F!=F) else F
    rel=r.get('pixels_changed',0)/max(a*384*384,1.0)
    return [F,np.log1p(rel),c,a,e,b,r['gate_prob']]
def load(fs):
    D=[L(f) for f in fs]; ids=sorted(set.intersection(*[set(d) for d in D])); return D,ids
def fit(fs):
    D,ids=load(fs); Xs=[];y=[]
    for d in D:
        for i in ids: Xs.append(X(d[i])); y.append(d[i]['prop']['iou']>d[i]['bb']['iou']+1e-9)
    m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)); m.fit(np.array(Xs),np.array(y)); return m
def ev(m,fs):
    D,ids=load(fs); res={k:[] for k in ['never','gateV','gate_sel','sel_all','gate_perfect']}
    per=[]
    for d in D:
        bb=np.array([d[i]['bb']['iou'] for i in ids]); pr=np.array([d[i]['prop']['iou'] for i in ids])
        gp=np.array([d[i]['gate_prob'] for i in ids]); acc=np.array([(d[i].get('F') is None) or (d[i]['F']!=d[i]['F']) or d[i]['F']>=0.5 for i in ids])
        p=m.predict_proba(np.array([X(d[i]) for i in ids]))[:,1]; esc=gp>=0.5
        v=dict(never=bb,gateV=np.where(esc&acc,pr,bb),gate_sel=np.where(esc&(p>=0.5),pr,bb),sel_all=np.where(p>=0.5,pr,bb),gate_perfect=np.where(esc,np.maximum(pr,bb),bb))
        per.append(v)
    out={k:float(np.mean([v[k].mean() for v in per])) for k in per[0]}
    # bootstrap CI of gate_sel - never and sel_all - never
    S={k:np.stack([v[k] for v in per]) for k in per[0]}; n=len(ids)
    for k in ['gate_sel','sel_all']:
        dd=[(S[k][:,ix]-S['never'][:,ix]).mean() for ix in (rng.integers(0,n,n) for _ in range(2000))]
        out[k+'_ci']=[float(x) for x in np.percentile(dd,[2.5,97.5])]
    out['n']=n; return out
out=[]
for bk in ['strong','segformer']:
    for v in ['medsam_box','medgemma4b_p1','qwen7b_p3']:
        vf=sorted(glob.glob(f'results_val/{bk}_s?__{v}.jsonl'))
        if not vf: continue
        m=fit(vf)
        for name,pat in [('Kvasir test',f'results_rev/{bk}_s?__{v}.jsonl')]+[(d,f'results_ext/{d}_{bk}_s?__{v}.jsonl') for d in ['CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB']]:
            fs=sorted(glob.glob(pat))
            if not fs: continue
            r=ev(m,fs); r.update(bk=bk,v=v,ds=name,seeds=len(fs)); out.append(r)
            print(f"{bk:9}{v:14}{name:18}{len(fs)} never {r['never']:.3f} gateV {r['gateV']:+.3f} gate+learnedAccept {r['gate_sel']-r['never']:+.4f} [{r['gate_sel_ci'][0]:+.3f},{r['gate_sel_ci'][1]:+.3f}] learnedOnly {r['sel_all']-r['never']:+.4f} [{r['sel_all_ci'][0]:+.3f},{r['sel_all_ci'][1]:+.3f}] gate+perfectAccept {r['gate_perfect']-r['never']:+.3f}".replace('gateV +','gateV ').replace(f"gateV {r['gateV']:+.3f}",f"gateV {r['gateV']-r['never']:+.3f}"))
json.dump(out,open("results_rev/learned_acceptance.json","w"),indent=1)

from scipy.stats import wilcoxon
print('---')
for bk in ['strong','segformer']:
    m=fit(sorted(glob.glob(f'results_val/{bk}_s?__medsam_box.jsonl')))
    print(bk,'coef [F,logrel,conf,area,entropy,boundary,gate]',np.round(m[-1].coef_[0],2))
    for name,pat in [('K',f'results_rev/{bk}_s?__medsam_box.jsonl')]+[(d,f'results_ext/{d}_{bk}_s?__medsam_box.jsonl') for d in ['CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB']]:
        if not glob.glob(pat): continue
        D,ids=load(sorted(glob.glob(pat))); G=[];N=[];acc=[];hlp=[]
        for d in D:
            bb=np.array([d[i]['bb']['iou'] for i in ids]); pr=np.array([d[i]['prop']['iou'] for i in ids]); gp=np.array([d[i]['gate_prob'] for i in ids])
            p=m.predict_proba(np.array([X(d[i]) for i in ids]))[:,1]; a=(gp>=0.5)&(p>=0.5)
            G.append(np.where(a,pr,bb)); N.append(bb); acc.append(a.mean()); hlp.append(((pr>bb)[a]).mean() if a.any() else np.nan)
        G=np.mean(G,0); N=np.mean(N,0)
        print(f"  {name:18} accepted {100*np.mean(acc):.1f}% of images, of which helped {100*np.nanmean(hlp):.0f}%  p={wilcoxon(G,N).pvalue:.2g}")
