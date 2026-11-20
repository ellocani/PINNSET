from viz_common import *
from fdm import solve
import os
R=pickle.load(open('R3.pkl','rb')); res=R['res']; ts=R['ts'].ravel(); th_s=R['th_s'].ravel()
Rb=pickle.load(open('R3b.pkl','rb'))['res'] if os.path.exists('R3b.pkl') else []
fig,ax=plt.subplots(3,1,figsize=(7.4,13.4),gridspec_kw=dict(hspace=0.5,height_ratios=[3,3.2,2.8]))
for Bi,c,lab in [(5,'#AE3EC9','Bi = 5'),(20,'#1C8574','Bi = 20 (정답)'),(80,'#F08C00','Bi = 80')]:
    tt=np.geomspace(t0*1.01,0.2,120); x,_,Y=solve(Bi=Bi,tend=0.2,ic='ones',teval=tt)
    ax[0].plot(tt*tref,TW+(TM-TW)*Y[0],color=c,lw=2,label=lab)
ax[0].plot(ts*tref,TW+(TM-TW)*th_s,'o',color='#333',ms=5,label='가상 센서 (잡음 0.5℃)')
ax[0].set_xscale('log'); ax[0].set_xlabel('시간 (초, 로그 눈금)'); ax[0].set_ylabel('캐비티 벽면 온도 (℃)')
ax[0].set_title('벽면 온도 센서가 있으면 접촉 열저항(Bi)이 갈린다',loc='left'); ax[0].legend(frameon=False,fontsize=9.3)
groups=[('센서 없음',[r for r in res if not r['data']],'#C92A2A'),('센서 있음\n기본 학습',[r for r in res if r['data']],'#74A9E0'),('센서 있음\n보강 학습',Rb,'#1971C2')]
for k,(name,q,c) in enumerate(groups):
    if not q: continue
    ax[1].scatter([k]*len(q),[r['Bi'] for r in q],s=70,color=c,zorder=3)
    lo,hi=min(r['Bi'] for r in q),max(r['Bi'] for r in q); Lm=np.median([r['loss'] for r in q])
    ax[1].text(k,hi*1.35,f"{lo:.1f}~{hi:.1f}\n손실 {Lm:.1e}",ha='center',fontsize=9)
ax[1].axhline(R['BI'],color='#1C8574',ls='--'); ax[1].text(2.45,R['BI']*1.06,'정답 20',color='#1C8574',fontsize=9.5,ha='right')
ax[1].set_yscale('log'); ax[1].set_ylim(2,60); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([g[0] for g in groups]); ax[1].set_xlim(-0.5,2.5)
ax[1].set_ylabel('PINN이 역산한 Bi (로그 눈금)'); ax[1].set_title('틀린 쪽이 손실은 더 낮고, 시드끼리도 모인다',loc='left')
for r in Rb:
    it,b=zip(*r['traj']); ax[2].plot(it,b,'o-',ms=3.5,label=f"시드 {r['seed']}")
ax[2].axhline(20,color='#1C8574',ls='--'); ax[2].set_xlabel('학습 스텝'); ax[2].set_ylabel('Bi 추정값'); ax[2].legend(frameon=False,fontsize=9)
ax[2].set_title('보강 학습 중 Bi 추정값의 움직임',loc='left')
fig.savefig('PINN_3_역문제와_식별가능성.png',dpi=190,bbox_inches='tight',facecolor='white'); print('saved')
