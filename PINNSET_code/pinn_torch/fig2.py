from viz_common import *
from fdm import solve, AG, AM, a_of
import pandas as pd, os
P=load('P2.pkl'); D=4
pn=lambda g:(np.asarray(g,float)-0.10)/0.30*2-1
xs=np.linspace(0,1,401)
thg_of=lambda Tw,Tm=TM:(105-Tw)/(Tm-Tw); thnf_of=lambda Tw,Tm=TM:(TNF-Tw)/(Tm-Tw)
out={}
# ---- FDM 대조 (비선형) : 여러 조건에서 FLR 비교
chk=[]
for Tw in (20,35,50,65,80):
    g=thg_of(Tw); nf=thnf_of(Tw); ts=np.array([1.5,4.45,8.0])/tref
    x,tt,Y=solve(thg=g,tend=ts.max()*1.01,teval=ts)
    for j,s in enumerate(ts):
        fp=flr(xs,pred(P,xs,s,pn(g),D),nf); ff=flr(x,Y[:,j],nf)
        ep=np.abs(pred(P,x,s,pn(g),D)-Y[:,j]).max()*(TM-Tw)
        chk.append((Tw,s*tref,ff,fp,ep))
C=pd.DataFrame(chk,columns=['Tw','t','FLR_FDM','FLR_PINN','최대오차℃'])
print(C.round(4).to_string(index=False))
print(f"FLR 절대오차: 중앙 {np.median(np.abs(C.FLR_FDM-C.FLR_PINN)):.4f}, 최대 {np.max(np.abs(C.FLR_FDM-C.FLR_PINN)):.4f}")
# ---- 라벨 운전점에서 물성 일정 vs 온도 의존
g0=thg_of(TW); nf0=thnf_of(TW)
f_const=flr(xs,series(xs,4.45/tref),nf0); f_nl=flr(xs,pred(P,xs,4.45/tref,pn(g0),D),nf0)
x,tt,Y=solve(thg=g0,tend=4.46/tref,teval=[4.45/tref]); f_fdm=flr(x,Y[:,0],nf0)
print(f"라벨 운전점 FLR @4.45s: 물성 일정 {f_const:.4f}, 온도 의존 PINN {f_nl:.4f}, 온도 의존 FDM {f_fdm:.4f}")
# ---- 지도
Tws=np.linspace(20,80,41); tfs=np.linspace(1,10,41); M=np.zeros((len(tfs),len(Tws))); Mc=np.zeros_like(M)
for j,Tw in enumerate(Tws):
    g=thg_of(Tw); nf=thnf_of(Tw)
    for i,tf in enumerate(tfs):
        M[i,j]=flr(xs,pred(P,xs,tf/tref,pn(g),D),nf); Mc[i,j]=flr(xs,series(xs,tf/tref),nf)
# ---- 실제 샷
ld=pd.read_csv(os.path.join(os.environ.get('PINNSET_DATA','.'),'labeled_data.csv'),parse_dates=['TimeStamp']).drop_duplicates(['TimeStamp','PART_FACT_SERIAL','PART_NAME'])
cn=ld[ld.PART_NAME.str.startswith('CN7')]; gg=cn.groupby('TimeStamp')
S=gg[['Filling_Time','Mold_Temperature_3','Mold_Temperature_4','Barrel_Temperature_1','Barrel_Temperature_2','Barrel_Temperature_3']].first()
S['Tw']=S[['Mold_Temperature_3','Mold_Temperature_4']].mean(axis=1); S['Tm']=S[['Barrel_Temperature_1','Barrel_Temperature_2','Barrel_Temperature_3']].mean(axis=1)
S['R']=gg.Reason.apply(lambda r:'+'.join(sorted(set(r.dropna()))) or '-'); S=S.reset_index(); S=S[S.Tm>265]
S['cls']=np.where(S.R=='초기허용불량','A',np.where(S.TimeStamp.between('2020-10-16 05:55','2020-10-16 05:58')&(S.R!='-'),'B',np.where(S.R=='가스','C',np.where(S.R=='-','정상','U'))))
fig,ax=plt.subplots(3,1,figsize=(7.4,14.2),gridspec_kw=dict(hspace=0.45,height_ratios=[2.6,3.2,4.6]))
Ts=np.linspace(20,280,300); ax[0].plot(Ts,[a_of((t-TW)/(TM-TW),g0)*AM for t in Ts],color='#1971C2',lw=2.2)
ax[0].axvline(105,color='#888',ls='--',lw=.8); ax[0].text(107,0.112,'유리전이 105℃',fontsize=9,color='#555')
ax[0].set_xlabel('온도 (℃)'); ax[0].set_ylabel('열확산율 (mm²/s)'); ax[0].set_title('온도 의존 열확산율 가정 (용융 0.080 → 유리 0.115 mm²/s)',loc='left')
ax[0].text(30,0.083,'유리 상태 0.115',fontsize=9.3); ax[0].text(200,0.086,'용융 상태 0.080',fontsize=9.3)
Cc=['#1971C2','#2F9E44','#F08C00']
for s,c in zip([1.5,4.45,8.0],Cc):
    x,tt,Y=solve(thg=g0,tend=s/tref*1.001,teval=[s/tref])
    ax[1].plot(x*h_mm,TW+(TM-TW)*Y[:,0],'-',color=c,lw=2.6,alpha=.35)
    ax[1].plot(xs[::25]*h_mm,TW+(TM-TW)*pred(P,xs[::25],s/tref,pn(g0),D),'o',color=c,ms=4.5)
    ax[1].plot(xs*h_mm,TW+(TM-TW)*series(xs,s/tref),':',color=c,lw=1.4)
    ax[1].text(h_mm*1.01,TW+(TM-TW)*Y[-1,0]+{1.5:9,4.45:-9}.get(s,0),f'{s:g}초',color=c,va='center',fontsize=9.5)
ax[1].axhline(TNF,color='#555',ls='--',lw=.8)
ax[1].set_xlim(0,h_mm*1.12); ax[1].set_xlabel('금형 벽에서의 거리 (mm)'); ax[1].set_ylabel('온도 (℃)')
ax[1].set_title('온도 의존 물성: 해석해 없음. PINN(점) = 수치해(굵은 선)',loc='left')
ax[1].legend(handles=[plt.Line2D([],[],marker='o',ls='',color='#333',label='PINN (온도 의존)'),plt.Line2D([],[],lw=2.6,alpha=.35,color='#333',label='FDM 수치해 (온도 의존)'),
                      plt.Line2D([],[],ls=':',color='#333',label='물성 일정 해석해')],loc='lower right',frameon=False,fontsize=9.3)
cs=ax[2].contourf(Tws,tfs,M,levels=np.linspace(0,0.5,21),cmap='Blues')
cl=ax[2].contour(Tws,tfs,M,levels=[0.1,0.15,0.2,0.25,0.3,0.35,0.4],colors='#0B3D91',linewidths=.8); ax[2].clabel(cl,fmt='%.2f',fontsize=8.5)
ax[2].contour(Tws,tfs,Mc,levels=[0.1,0.15,0.2,0.25,0.3,0.35,0.4],colors='#C92A2A',linewidths=.8,linestyles=':')
for k,(c,m,sz,lab) in {'정상':('#555','.',6,'정상 샷'),'C':('#1C8574','o',26,'C 캐비티'),'A':('#E03131','s',40,'A 과도응답'),'B':('#F08C00','^',46,'B 고속')}.items():
    q=S[S.cls==k]; ax[2].scatter(q.Tw,q.Filling_Time,s=sz,c=c,marker=m,edgecolor='white' if k!='정상' else 'none',linewidth=.6,label=f'{lab} ({len(q)})',zorder=3)
ax[2].set_xlabel('금형온도 (℃)'); ax[2].set_ylabel('충진시간 (초)')
ax[2].set_title('성형 윈도우 지도: 조건 전체의 FLR을 신경망 하나로',loc='left')
ax[2].legend(loc='upper right',fontsize=9,frameon=True,framealpha=.9)
ax[2].text(0.0,-0.17,'색·파란 선 = PINN (온도 의존 물성).  빨간 점선 = 물성 일정 해석해.  용융온도 275.5℃ 고정',transform=ax[2].transAxes,fontsize=8.8,color='#555')
axin=ax[2].inset_axes([0.60,0.30,0.37,0.30])
axin.contourf(Tws,tfs,M,levels=np.linspace(0,0.5,41),cmap='Blues')
for k,(c,m,sz) in {'정상':('#555','.',8),'C':('#1C8574','o',22),'A':('#E03131','s',30),'B':('#F08C00','^',34)}.items():
    q=S[S.cls==k]; axin.scatter(q.Tw,q.Filling_Time,s=sz,c=c,marker=m,zorder=3)
axin.set_xlim(20,27); axin.set_ylim(3.9,8.6); axin.tick_params(labelsize=8); axin.set_title('실제 운전 영역 확대',fontsize=8.8)
fig.savefig('PINN_2_비선형물성과_성형윈도우지도.png',dpi=190,bbox_inches='tight',facecolor='white')
print('saved')
