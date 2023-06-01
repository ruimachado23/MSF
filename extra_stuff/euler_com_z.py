import numpy as np
import matplotlib.pyplot as plt
import math

#parametros
g=9.80

deltat= 0.01
t0=0
tf=20

x0=0
vx0=(98.5/3.6)    #100*cos(10)
ax0=0

y0=0
vy0=(17.4/3.6)    #100*sin(10)
ay0= -g

z0=0
vz0=0
az0=0


vterminal=(100/3.6)


#inicialização
n=np.int_((tf-t0)/deltat)

t=np.zeros(n+1)             # n+1 elementos; último índice n
t[0]=t0   

y=np.zeros(n+1)
vy=np.zeros(n+1)
ay=np.zeros(n+1)
y[0]=y0
vy[0]=vy0
ay[0]=ay0

x=np.zeros(n+1)
vx=np.zeros(n+1)
ax=np.zeros(n+1)
x[0]=x0
vx[0]=vx0
ax[0]=ax0

z=np.zeros(n+1)
vz=np.zeros(n+1)
az=np.zeros(n+1)
z[0]=z0
vz[0]=vz0
az[0]=az0

axr=np.zeros(n+1)
ayr=np.zeros(n+1)
azr=np.zeros(n+1)
axr[0]=ax0
ayr[0]=ay0
azr[0]=az0

xr=np.zeros(n+1)
yr=np.zeros(n+1)
zr=np.zeros(n+1)
xr[0]=x0
yr[0]=y0
zr[0]=z0

vxr=np.zeros(n+1)
vyr=np.zeros(n+1)
vzr=np.zeros(n+1)
vxr[0]=vx0
vyr[0]=vy0
vzr[0]=az0


# Método de Euler (n+1 elementos)

i=0
while y[i] >= 0:
    ay[i+1] = -g
    ax[i+1] = 0
    az[i+1] = 0

    z[i+1]=z[i]+vz[i]*deltat+(1/2*az[i]*deltat**2)
    vz[i+1]=vz[i] + az[i]*deltat

    y[i+1]=y[i]+vy[i]*deltat+(1/2*ay[i]*deltat**2)
    vy[i+1]=vy[i] + ay[i]*deltat
    
    x[i+1]=x[i]+vx[i]*deltat+(1/2*ax[i]*deltat**2)
    vx[i+1]=vx[i] + ax[i]*deltat 
    
    t[i+1]=t[i]+ deltat

    axr[i+1] =ax[i] - (g/vterminal**2 * math.sqrt(vxr[i]**2+vyr[i]**2+vzr[i]**2) *vxr[i])
    ayr[i+1] =ay[i] - (g/vterminal**2 * math.sqrt(vxr[i]**2+vyr[i]**2+vzr[i]**2) *vyr[i])
    azr[i+1] =az[i] - (g/vterminal**2 * math.sqrt(vxr[i]**2+vyr[i]**2+vzr[i]**2) *vzr[i])

    xr[i+1]=xr[i]+vxr[i]*deltat+(1/2*axr[i]*deltat**2)
    yr[i+1]=yr[i]+vyr[i]*deltat+(1/2*ayr[i]*deltat**2)
    zr[i+1]=zr[i]+vzr[i]*deltat+(1/2*azr[i]*deltat**2)

    vxr[i+1]=vxr[i] + axr[i]*deltat 
    vyr[i+1]=vyr[i] + ayr[i]*deltat
    vzr[i+1]=vzr[i] + azr[i]*deltat

    i+=1


plt.xlim([0,30])
plt.ylim([0,2])

plt.plot(xr,yr)

plt.show()