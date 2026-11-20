from viz_common import *
P=load('P1.pkl'); D=4
xs=np.linspace(0,1,801)
thnf=(TNF-TW)/(TM-TW)
fig,ax=plt.subplots(3,1,figsize=(7.4,13.2),gridspec_kw=dict(hspace=0.42))
# (a) 온도 분포
C=['#1971C2','#2F9E44','#F08C00','#C92A2A']
for (sec,c) in zip([1.0,4.45,15.0,37.8],C):
    t=sec/tref
    ax[0].plot(xs*h_mm,TW+(TM-TW)*series(xs,t),'-',color=c,lw=2.4,alpha=.35)
    ax[0].plot(xs[::40]*h_mm,TW+(TM-TW)*pred(P,xs[::40],t,D=D),'o',color=c,ms=4.5)
    ax[0].plot(xs*h_mm,TW+(TM-TW)*erf_semi(xs,t),':',color=c,lw=1.4)
    ax[0].text(h_mm*1.01,TW+(TM-TW)*series(np.array([1.0]),t)[0]+{1.0:9,4.45:-9}.get(sec,0),f'{sec:g}초',color=c,va='center',fontsize=9.5)
ax[0].axhline(TNF,color='#555',lw=.8,ls='--'); ax[0].text(0.02,TNF+5,'무유동 온도 110℃',fontsize=9,color='#555')
ax[0].set_xlabel('금형 벽에서의 거리 (mm)  →  오른쪽 끝이 두께 중심'); ax[0].set_ylabel('온도 (℃)'); ax[0].set_xlim(0,h_mm*1.12)
ax[0].set_title('검증: PINN(점)이 정확해(굵은 선)를 재현한다',loc='left')
ax[0].legend(handles=[plt.Line2D([],[],marker='o',ls='',color='#333',label='PINN'),plt.Line2D([],[],lw=2.4,alpha=.35,color='#333',label='정확해 (유한 두께)'),
                      plt.Line2D([],[],ls=':',color='#333',label='반무한 해석해 (erf)')],loc='lower right',frameon=False,fontsize=9.5)
# (b) 오차
secs=np.geomspace(0.15,38,60); err=[np.abs(pred(P,xs,s/tref,D=D)-series(xs,s/tref)).max()*(TM-TW) for s in secs]
erf_err=[np.abs(erf_semi(xs,s/tref)-series(xs,s/tref)).max()*(TM-TW) for s in secs]
ax[1].semilogx(secs,err,'o-',color='#1971C2',ms=3.5,label='PINN의 오차'); ax[1].semilogx(secs,erf_err,'-',color='#C92A2A',label='반무한 가정의 오차')
ax[1].axvline(4.45,color='#888',ls='--',lw=.8); ax[1].text(4.6,max(erf_err)*0.9,'정상 충진 종료',fontsize=9,color='#555')
ax[1].set_xlabel('시간 (초, 로그 눈금)'); ax[1].set_ylabel('최대 온도 오차 (℃)'); ax[1].legend(frameon=False,fontsize=9.5)
ax[1].set_title('정확해 대비 최대 오차: PINN 대 반무한 가정',loc='left')
# (c) FLR
F_ex=[flr(xs,series(xs,s/tref),thnf) for s in secs]; F_p=[flr(xs,pred(P,xs,s/tref,D=D),thnf) for s in secs]; F_e=[flr(xs,erf_semi(xs,s/tref),thnf) for s in secs]
ax[2].plot(secs,F_ex,'-',lw=3,alpha=.35,color='#333',label='정확해'); ax[2].plot(secs,F_p,'o',ms=3.5,color='#1971C2',label='PINN'); ax[2].plot(secs,F_e,':',color='#C92A2A',lw=1.6,label='반무한 해석해')
for s,l in [(4.45,'정상'),(8.27,'초기허용 1샷'),(37.8,'냉각 종료')]: ax[2].axvline(s,color='#aaa',lw=.8,ls='--'); ax[2].text(s+0.4,0.03,l,fontsize=9,color='#555',rotation=90,va='bottom')
ax[2].set_xlabel('시간 (초)'); ax[2].set_ylabel('FLR (고화층 두께 / 반두께)'); ax[2].legend(frameon=False,fontsize=9.5,loc='upper left')
ax[2].set_title('고화층 비율: 반무한 가정이 틀어지는 시점',loc='left')
fig.savefig('PINN_1_검증과_유한두께.png',dpi=190,bbox_inches='tight',facecolor='white')
i=np.argmin(np.abs(secs-4.45)); j=np.argmin(np.abs(secs-37.8))
print(f"PINN 최대 오차: 전 구간 최대 {max(err):.2f}℃, 중앙 {np.median(err):.2f}℃")
print(f"FLR @4.45s: 정확해 {F_ex[i]:.4f}, PINN {F_p[i]:.4f}, 반무한 {F_e[i]:.4f}")
print(f"FLR @37.8s: 정확해 {F_ex[j]:.4f}, PINN {F_p[j]:.4f}, 반무한 {F_e[j]:.4f}")
k=next(n for n,(a,b) in enumerate(zip(F_ex,F_e)) if abs(a-b)/max(a,1e-9)>0.01)
print(f"반무한 가정의 FLR 오차가 1%를 넘는 시점: {secs[k]:.1f}초")
