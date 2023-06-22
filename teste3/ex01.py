# corpo a mover-se num oscilador quártico

import numpy as np
from matplotlib import pyplot as plt

#alínea a) - gráfico da energia potencial num intervalo limitado

m = 0.5
xeq = 0
k = 2
alpha = -0.1
beta = 0.02
x = np.arange(-4, 4, 0.1)
Ep = 0.5*k*(x**2)+alpha*(x**3)-beta*(x**4)

plt.plot(x,Ep)
plt.grid()
plt.show()

#alínea b) - calcular a lei do movimento para um x0 e v0 | calcular a energia mecânica

def oscHarmSimp_1D(x0,v0,k,m,n,dt,alpha,beta):
    x=np.empty(n+1)
    v=np.empty(n+1)
    a=np.empty(n+1)
    Em = np.empty(n+1)
    x[0]=x0
    v[0]=v0
    for i in range(n):
        a[i]=(-k/m*x[i]) - ((3/m*alpha)*(x[i]**2)) + ((4/m*beta)*(x[i]**3))
        v[i+1]=v[i]+a[i]*dt
        x[i+1]=x[i]+v[i+1]*dt
        Em[i] = 0.5*m*(v[i]**2) + (0.5*k*(x[i]**2) + alpha*x[i]**3 - beta*x[i]**4)
    
    Em[n] = 0.5*m*(v[n]**2) + (0.5*k*(x[n]**2) + alpha*x[n]**3 - beta*x[n]**4)

    return x,v,a,Em

x0 = 1.5
v0 = 0.5
ti = 0
tf = 20
dt = 0.001
n = int((tf-ti)/dt)

t = np.linspace(ti, tf, n+1)

values = oscHarmSimp_1D(x0,v0,k,m,n,dt,alpha,beta)
x = values[0]
Em = values[3]

plt.plot(t,x)
plt.plot(t,Em)
plt.grid()
plt.show()

#alínea c) - Limites onde se efetua o movimento, a frequencia e o periodo? | Apresentar o resultado com a precisão de 4 algarismos

import numpy as np
import matplotlib.pyplot as plt


def maximo(xm1, xm2, xm3, ym1, ym2, ym3):  # máximo pelo polinómio de Lagrange
    xab = xm1-xm2
    xac = xm1-xm3
    xbc = xm2-xm3

    a = ym1/(xab*xac)
    b = -ym2/(xab*xbc)
    c = ym3/(xac*xbc)

    xmla = (b+c)*xm1+(a+c)*xm2+(a+b)*xm3
    xmax = 0.5*xmla/(a+b+c)

    xta = xmax-xm1
    xtb = xmax-xm2
    xtc = xmax-xm3

    ymax = a*xtb*xtc+b*xta*xtc+c*xta*xtb
    return xmax, ymax

x = np.empty(n+1)
v = np.empty(n+1)
a = np.empty(n+1)
Em = np.empty(n+1)
Ep = np.empty(n+1)
x[0] = x0
v[0] = v0

countMaximos = 0
maxTotal = 0
difTempos = []
maximos = []

for i in range(n):
    a[i] = (-k/m*x[i]) - ((3/m*alpha)*(x[i]**2)) + ((4/m*beta)*(x[i]**3))

    v[i+1] = v[i] + a[i]*dt

    x[i+1] = x[i] + v[i+1]*dt

    Ep[i] = 0.5*k*(x[i]**2) + alpha*x[i]**3 - beta*x[i]**4

    Em[i] = Ep[i] + 0.5*m*v[i]**2

Ep[n] = 0.5*k*(x[n]**2) + alpha*x[n]**3 - beta*x[n]**4
Em[n] = Ep[n]+0.5*m*v[n]**2


Amp = []
AmpNeg = []

tempos = []
periodos = []
freq = []

for i in range(n):
    if (x[i-1] < x[i] > x[i+1] and i > 0):
        Amp.append(x[i])
        tempos.append(t[i])

for i in range(n):
    if (x[i-1] > x[i] < x[i+1] and i > 0):
        AmpNeg.append(x[i])


for i in range(1, len(tempos)-1):
    periodos.append(tempos[i+1]-tempos[i])
    freq.append(1/periodos[i-1])

A = sum(Amp)/(len(Amp))
T = sum(periodos)/(len(periodos))

print("Amplitude+:")
print("{:0.4f},{:0.4f}".format(Amp[0], Amp[-1]))
print("Amplitude-:")
print("{:0.4f},{:0.4f}".format(AmpNeg[0], AmpNeg[-1]))
print("Periodo:")
print("{:0.4f},{:0.4f}".format(periodos[0], periodos[-1]))
print("Frequencia:")
print("{:0.4f},{:0.4f}".format(freq[0], freq[-1]))

#alínea d) - Fazer a análise de Fourier da solução encontrada

import numpy as np
import matplotlib.pyplot as plt

ti_estac = 0

def fourier(x0,v0,m,k,alpha,t,n,ti_estac, dt,beta): 
    #faz a preparação necessária ao cálculo dos coeficientes de Fourier e apresenta-os num gŕafico de barras
    #atenção: alterar a equação da aceleração se necessário
    x=np.empty(n+1)
    v=np.empty(n+1)
    a=np.empty(n+1)
    E=np.empty(n+1)
    
    v[0]=v0
    x[0]=x0
    cntMax=0
    ind=np.transpose([0 for i in range(1000)])
    afo=np.zeros(15)
    bfo=np.zeros(15)


    for i in range(n):
        a[i]=(-k/m*x[i]) - ((3/m*alpha)*(x[i]**2)) + ((4/m*beta)*(x[i]**3))
        v[i+1]=v[i]+a[i]*dt
        x[i+1]=x[i]+v[i+1]*dt
        E[i]=0.5*m*v[i]**2+0.5*k*x[i]**2
        if t[i]>ti_estac and x[i-1] < x[i] and  x[i+1] < x[i]:
            cntMax+=1
            ind[cntMax]=int(i)
    		
    t0=ind[cntMax-1]
    t1=ind[cntMax]    
    for i in range(15):
        af, bf=fourier_calc(t,x,t0,t1,dt,i)
        afo[i]=af
        bfo[i]=bf

    ii=np.linspace(0,14,15)
    plt.figure()
    plt.ylabel('| a_n |')
    plt.xlabel('n')
    plt.bar(ii,np.abs(afo))
    plt.grid()
    plt.show()

    ii=np.linspace(0,14,15)
    plt.figure()
    plt.ylabel('| b_n |')
    plt.xlabel('n')
    plt.bar(ii,np.abs(bfo))
    plt.grid()
    plt.show()

    ii=np.linspace(0,14,15)
    plt.figure()
    plt.bar(ii,np.sqrt(np.abs(afo)**2 + np.abs(bfo)**2))
    plt.grid()
    plt.show()

def fourier_calc(t,x,t0,t1,dt,nf):
    #faz os cálculos dos coeficientes de Fourier. Parte de 'fourier'
    T=t[t1]-t[t0]
    ome=2*np.pi/T

    s1=x[t0]*np.cos(nf*ome*t[t0])
    s2=x[t1]*np.cos(nf*ome*t[t1])
    st=x[t0+1:t1]*np.cos(nf*ome*t[t0+1:t1])
    soma=np.sum(st)
    
    q1=x[t0]*np.sin(nf*ome*t[t0])
    q2=x[t1]*np.sin(nf*ome*t[t1])
    qt=x[t0+1:t1]*np.sin(nf*ome*t[t0+1:t1])
    somq=np.sum(qt)
    
    intega=((s1+s2)/2+soma)*dt
    af=2/T*intega
    integq=((q1+q2)/2+somq)*dt
    bf=2/T*integq
    return af,bf

t = np.linspace(ti, tf, n+1)

fourier(x0,v0,m,k,alpha,t,n,ti_estac, dt,beta)