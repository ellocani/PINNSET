import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10.5,'axes.titlesize':12.5,'axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False})
TW, TM, TNF = 22.8, 275.5, 110.0
def load(fn):
    return {k:np.asarray(v,float) for k,v in pickle.load(open(fn,'rb')).items()}
def forward(P, F, D):
    """학습된 가중치로 신경망 값만 계산 (tanh 은닉층 D개 → 선형 출력). 학습은 pinn_torch_train.py(PyTorch)"""
    h=np.tanh(F@P['W0']+P['b0'])
    for l in range(1,D): h=np.tanh(h@P[f'W{l}']+P[f'b{l}'])
    return h@P['Wo']+P['bo']
def pred(P, x, t, p=None, D=4):
    x=np.asarray(x,float).reshape(-1,1); t=np.asarray(t,float).reshape(-1,1)
    t=np.broadcast_to(t,x.shape).copy() if t.size==1 else t
    pp=None if p is None else np.broadcast_to(np.asarray(p,float).reshape(-1,1),x.shape).copy()
    return forward(P,feats(x,t,pp)[0],D).ravel()
def flr(xgrid, th, thnf):
    if not (th>=thnf).any(): return 1.0          # 두께 전체가 굳음
    i=np.argmax(th>=thnf)
    return np.interp(thnf,[th[i-1],th[i]],[xgrid[i-1],xgrid[i]]) if i>0 else 0.0
