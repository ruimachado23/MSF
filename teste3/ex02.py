# corpo a mover-se num oscilador forçado

import numpy as np
from matplotlib import pyplot as plt

#alínea a) - calcular a amplitude da oscilação no regime estacionário, de um oscilador harmónico

x0 = 0
b = 0.05
k =1
m = 1
w_f = 1.4
F0 = 7.5
x0 = 2
v0 = 4
ti = 0
tf = 400
dt = 0.01
n = int((tf-ti)/dt)

def oscSimpFA_1D(x0,v0,k,m,t,b,F0,w_f,n,dt):
    #oscilador Simples sujeito forçado e amortecido
    x=np.empty(n+1)
    v=np.empty(n+1)
    a=np.empty(n+1)
    x[0]=x0
    v[0]=v0
    for i in range(n):
        a[i]=-k/m*x[i]+(-b*v[i]+F0*np.cos(w_f*t[i]))/m
        v[i+1]=v[i]+a[i]*dt
        x[i+1]=x[i]+v[i+1]*dt
    return x,v,a

def amp_per_comp(x, t, n, reg_est):
    #amplitde, periodo e comprimeno de onda com regime estacionario
    ind_max=[i for i in range(1,n-1) if x[i-1]<=x[i]>=x[i+1] if t[i]>reg_est]
    x_max=[x[i] for i in ind_max]
    t_max=[t[i] for i in ind_max]
    A=np.average(x_max)

    T_lst=[t_max[i+1]-t_max[i] for i in range(len(t_max)-1)]
    T=np.average(T_lst)

    lmbd_lst=[x_max[i+1]-x_max[i] for i in range(len(x_max)-1)]
    lmbd=np.average(lmbd_lst)

    return A, T, lmbd

t = np.linspace(ti,tf,n+1)
values = oscSimpFA_1D(x0,v0,k,m,t,b,F0,w_f,n,dt)
x = values[0]

reg_est = 250
values1 = amp_per_comp(x, t, n, reg_est)
A = values1[0]

print("Amplitude: ", A)
plt.plot(t, x)
plt.grid()
plt.show()

#alínea b) - calcular a amplitude da oscilação no regime estacionário, de um oscilador quártico

import numpy as np
import matplotlib.pyplot as plt

k = 1
m = 1
b = 0.05
F0 = 7.5
Wf = 1.4
alpha = 0.001

def oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf):
    ti = 0
    n = int((tf-ti)/dt+0.1)
    tempo = np.empty(n+1)
    x = np.empty(n+1)
    vx = np.empty(n+1)
    a = np.empty(n+1)
    Em = np.empty(n+1)

    t0 = 0.
    x0 = 2.0
    vx0 = 4

    tempo[0] = t0
    vx[0] = vx0
    x[0] = x0

    ampl = 0
    countMax = 0
    tMax = []
    periodo = []
    for i in range(n):
        tempo[i+1] = tempo[i]+dt
        a[i] = -(k/m)*x[i]*(1+2*alpha*x[i]**2) - \
            (b/m)*vx[i]+(F0/m)*np.cos(Wf*tempo[i])
        vx[i+1] = vx[i]+a[i]*dt
        x[i+1] = x[i]+vx[i+1]*dt

    return a, vx, x, tempo


dt = 0.001
tf = 300
a_1, vx_1, x_1, t_1 = oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf)
dt = 0.01
a_2, vx_2, x_2, t_2 = oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf)

x_temp = x_1[t_1 > 150]
t_temp = t_1[t_1 > 150]
maximos_x = x_temp[:-2][np.diff(np.sign(np.diff(x_temp))) == -2]
maximos_t = t_temp[:-2][np.diff(np.sign(np.diff(x_temp))) == -2]
print("Amplitude:", np.round(np.mean(maximos_x), 3), "m")
print("Período:",  np.round(np.mean(np.diff(maximos_t)), 3), "s")


#alínea c) - calcular a uma nova amplitude da oscilação no regime estacionário, de um oscilador quártico
#            no instante 400 s, a frequência da força externa é mudada de 1.4 rad/s para 1.37rad/s.

import numpy as np
import matplotlib.pyplot as plt


b = 0.05
F0 = 7.5
Wf = 1.37
alpha = 0.001
dt = 0.001
tf = 600
dt = 0.01
n = int(tf/dt+0.1)
t = np.empty(n+1)

def oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf,n,t):
    
    x = np.empty(n+1)
    vx = np.empty(n+1)
    a = np.empty(n+1)
    Em = np.empty(n+1)

    t0 = 0.
    x0 = 2.0
    vx0 = 4

    t[0] = t0
    vx[0] = vx0
    x[0] = x0

    
    ampl = 0
    countMax = 0
    tMax = []
    periodo = []
    for i in range(n):
        t[i+1] = t[i]+dt
        a[i] = -(k/m)*x[i]*(1+2*alpha*x[i]**2) - \
            (b/m)*vx[i]+(F0/m)*np.cos(Wf*t[i])
        vx[i+1] = vx[i]+a[i]*dt
        x[i+1] = x[i]+vx[i+1]*dt

    return a, vx, x, t



a_1, vx_1, x_1, t_1 = oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf,n,t)

a_2, vx_2, x_2, t_2 = oscilador_quartico(k,m,b,F0,Wf,alpha,dt, tf,n,t)
x_temp = x_1[t_1 > 400]
t_temp = t_1[t_1 > 400]
maximos_x = x_temp[:-2][np.diff(np.sign(np.diff(x_temp))) == -2]
maximos_t = t_temp[:-2][np.diff(np.sign(np.diff(x_temp))) == -2]
print("Amplitude:", np.round(np.mean(maximos_x), 3), "m")
print("Período:",  np.round(np.mean(np.diff(maximos_t)), 3), "s")