import numpy as np
import matplotlib.pyplot as plt

m = 60+12 # massa do corpo em kg
P = 0.48*735.4975 # potência em Cv -> W
x0 = 0 # posição inicial m
v0 = 0.5 # velocidade inicial m/s
u = 0.01 # coeficiente de resistência do alcatrão
cres = 0.9 # coeficiente de resistência do ar
A = 0.5 # Area frontal m²

ti = 0
tf = 500
dt = 0.001
n = int((tf-ti)/dt)

t = np.linspace(ti,tf,n+1)

def planoInclinado_res_1D(x0,v0,n,dt,cres,u,A,m,P):
    x=np.empty(n+1)
    vx=np.empty(n+1)
    ax=np.empty(n+1)
    
    p_ar=1.225
    g=9.8
    
    x[0]=x0
    vx[0]=v0
    ax[0]=0
    
    for i in range(n):
        if x[i] <= 1500:
            vv=np.abs(vx[i])
            ax[i]=-g*np.sin(np.radians(4)) -u*g*np.cos(np.radians(4)) -(0.5*cres*A*p_ar*vx[i]*vv)/m + P/(m*vx[i])
            vx[i+1]=vx[i]+ax[i]*dt
            x[i+1]=x[i]+vx[i]*dt
        else:
            vv=np.abs(vx[i])
            ax[i]=g*np.sin(np.radians(1)) -u*g*np.cos(np.radians(1)) -(0.5*cres*A*p_ar*vx[i]*vv)/m + P/(m*vx[i])
            vx[i+1]=vx[i]+ax[i]*dt
            x[i+1]=x[i]+vx[i]*dt
    return x,vx,ax

values = planoInclinado_res_1D(x0,v0,n,dt,cres,u,A,m,P)
x = values[0]

for i in range(n):
    if (2000-0.01)<x[i]<(2000+0.01):
        print("Tempo:", t[i], "s")
        break