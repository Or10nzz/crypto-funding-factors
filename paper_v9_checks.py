"""Checks quoted in the paper that the numbered scripts do not print (v3.5, 2026-11-24): funding/spot ratio intervals, LV 7-day i.i.d. t, sample halves and years, no-liquidation funding, FUND by year. Run after paper_v5."""
import json, math, numpy as np
from collections import Counter, defaultdict
from bisect import bisect_right
from datetime import datetime, timezone
import paper_v3 as v, paper_v5 as p5, paper_v6 as p6
from paper_ic_ls import D, ROOT, mean, fmt
from paper_funding import spot_opens
from paper_v7 import iid_t
p5.install(); data=p5.load5(); hv=v.hour_volume(data)
smap=json.loads((ROOT/"results/paper/spot_map.json").read_text()); spot={p:spot_opens(s) for p,(s,m) in smap.items()}
W,_=p5.build5(data,hv,spot,smap); yr=lambda t:datetime.fromtimestamp(t/1000,timezone.utc).year
print("weeks",len(W),fmt(W[0]["day"]),fmt(W[-1]["day"])); assert len(W)==317 and fmt(W[-1]["day"])=="2026-10-19"
P=lambda x:f"{x*100:+.3f}%"
def byyear(rows,k):
    g=defaultdict(list)
    for r in rows: g[yr(r["day"])].append(r[k])
    return {y:P(mean(x)) for y,x in sorted(g.items())}
fm=p6.ls_floor(W,v.plain("FUND"),6); fx=p6.ls_floor(W,v.plain("FUND"),6,floor=None,use_liq=False)
for l,r in(("main",fm),("cross",fx)):
    for k in("price","fund","gross","net"): print("b FUND",l,k,byyear(r,k))
f=np.array([-r[1] for w in W for r in w["ret"].values()])*100
print("c funding paid by long: mean %.3f median %.3f p1 %.2f p99 %.2f p0.1 %.2f p99.9 %.2f skew %.2f"%(f.mean(),np.median(f),*np.percentile(f,[1,99,.1,99.9]),((f-f.mean())**3).mean()/f.std()**3))
for key in("LV","LV14","LV30","MOM"):
    S_,P_,T=v.spot_rows(W,key); F=np.array([r["fund"] for r in T]); n=len(F)
    cut=int(datetime(2023,9,1,tzinfo=timezone.utc).timestamp()*1000)
    h=[np.array([r["fund"] for r in T if (r["day"]<cut)==b]) for b in(True,False)]
    Px=p6.spot_rows_x(W,key)
    print("d",key,"n",n,P(F.mean()),"iid %.2f"%iid_t(list(F)),"halves",[(len(a),P(a.mean()),"%.2f"%v.nw(a)[1]) for a in h],"years",byyear(T,"fund"))
    s0=np.array([r["net"] for r in S_]); s1=np.array([r["net"] for r in Px[0]])
    for lab,s in(("floored spot(Table4)",s0),("v6 spot",s1)):
        lo,hi=v.boot(lambda i:-F[i].mean()/s[i].mean(),n); print("e",key,lab,P(s.mean()),"ratio %.0f%% [%.0f%%,%.0f%%]"%(-100*F.mean()/s.mean(),100*lo,100*hi))
    print("h",key,"perp price gross",P(mean(r["gross"] for r in P_)),"net",P(mean(r["net"] for r in P_)))
    Wok=[]
    def okw(w): return {s:w["ret"][s] for s in w["spot"] if s in w["ret"] and s in w["sig"][key]}
    Ws=[w for w in W if len(okw(w))>=24]; pick=lambda w,R:v.top_bottom(w["sig"][key],6,R)
    nl=p6.ls_floor(Ws,pick,6,retkey=okw,floor=None,use_liq=False); a=np.array([r["fund"] for r in nl]); print("d2",key,"no liquidation funding",P(a.mean()),"%.2f"%v.nw(a)[1])
IV={s:p5.interest(c) for s,c in data.items()}
for w in W:
    w["exx"]={}
    for s in w["ret"]:
        c=data[s];d=w["day"]
        if c["times"] and c["times"][0]<=d-14*D: w["exx"][s]=p5.excess_funding(c,IV[s],d-7*D,d,False)
for lab,fn in(("price",lambda w,s:min(w["ret"][s][0],10)),("total",lambda w,s:min(sum(w["ret"][s]),10))):
    g=defaultdict(list)
    for w in W:
        ss=sorted([s for s in w["exx"] if w["exx"][s] is not None],key=lambda s:(w["exx"][s],s))
        if len(ss)<25: continue
        q=np.array_split(np.array(ss,dtype=object),5); g[yr(w["day"])].append(mean(fn(w,s) for s in q[0])-mean(fn(w,s) for s in q[4]))
    al=[x for a in g.values() for x in a]; print("f Q1-Q5",lab,{y:P(mean(x)) for y,x in sorted(g.items())},"all",P(mean(al)),"t4 %.2f t26 %.2f"%(v.nw(al)[1],p5.nwL(al,26)[1]),"ex2026",P(mean(x for y,a in g.items() if y!=2026 for x in a)))
cnt=defaultdict(Counter)
for s,c in data.items():
    t=c["times"]
    for k in range(1,len(t)):
        h_=round((t[k]-t[k-1])/3600000)
        if h_ in(1,2,4,8): 
            r=round((c["pre"][k+1]-c["pre"][k])*1e6)/100   # in bp
            cnt[(h_,yr(t[k]))][r]+=1
for k in sorted(cnt): n=sum(cnt[k].values()); print("g interval",k[0],"h",k[1],"n",n,"top rates bp",[(r,f"{100*x/n:.0f}%") for r,x in cnt[k].most_common(3)])
Wt,thr=v.drop_extreme(W,1)
for lab,q in(("simple",lambda X:p5.price_part_simple(X,lambda w:v.top_bottom(w["sig"]["MOM"],6,w["ret"]),lambda w:w["ret"])),("main",lambda X:mean(r["price"] for r in p5.ls5(X,v.plain("MOM"),6)))): print("i MOM price",lab,P(q(W)),P(q(Wt)),"drop",P(q(W)-q(Wt)))
m=p5.ls5(W,v.plain("MOM"),6); l=p5.ls5(W,v.plain("LV"),6); L={r["day"]:r for r in l}; ov=[0,0];ot=[0,0]
for w in W:
    d=w["day"]; a=L[d]; b=next((r for r in m if r["day"]==d))
    for s,sd,cp,cf in a["legs"]:
        if sd==-1:
            c=data[s];t1=p5.T1.get((s,d),d+7*D);i,j=bisect_right(c["times"],d),bisect_right(c["times"],t1);p0=c["bars"][d][0]
            dv=sum(IV[s][k]*((c["fpx"][k] or p0)/p0) for k in range(i,j))/12
            x=s in b["lo"]; ov[x]+=cf; ot[x]+=cf-dv
print("j LV short funding raw other/overlap",[P(x/317) for x in ov],"excess(default full week)",[P(x/317) for x in ot])
