#113765 Rui Machado

#determinar a evolução temporal da posição e da velocidade de um carro

#import de bibliotecas
import numpy as np
import matplotlib.pyplot as plt

m = 2000        #massa do corpo em kg
P = 40 * 1000   #é necessário converter a potência para W
x0 = 0          #posição inicial
v0 = 1          #velocidade inicial
u = 0.04        #coeficiente de resistência de rolamento
cres = 0.25     #coeficiente de resistência do ar
A = 2           #area frontal

ti = 0
tf = 300
dt = 0.001
n = int((tf-ti)/dt)

t = np.linspace(ti,tf,n+1)

def planoInclinado_res_1D(x0,v0,n,dt,cres,u,A,m,P,ang=0):
    x=np.empty(n+1)
    vx=np.empty(n+1)
    ax=np.empty(n+1)
    
    p_ar=1.225
    g=9.8
    
    x[0]=x0
    vx[0]=v0
    ax[0]=0
    
    for i in range(n):
        vv=np.abs(vx[i])
        ax[i]=-g*np.sin(ang) -u*g*np.cos(ang) -(0.5*cres*A*p_ar*vx[i]*vv)/m + P/(m*vx[i])
        vx[i+1]=vx[i]+ax[i]*dt
        x[i+1]=x[i]+vx[i]*dt
    return x,vx,ax

values = planoInclinado_res_1D(x0,v0,n,dt,cres,u,A,m,P,ang=0)
x = values[0]
v = values[1]

#gráfico de v em função de t
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.plot(t,v)
plt.grid()
plt.title("Evolução da velocidade em função do tempo")
plt.show()

#gráfico de x em função de t
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.plot(t,x)
plt.title("Evolução da posição em função do tempo")
plt.show()


