# Rui Machado 113765

import numpy as np
from matplotlib import pyplot as plt

# alínea a)

m = 1
k = 1
alpha = 0.05
x = np.arange(-8, 4, 0.1)               # limitaçao do intervalo
Ep = 0.5*k*(x**2)+alpha*(x**3)          # energia potencial 

plt.plot(x,Ep)
plt.grid()
plt.show()

# Resposta:
# Quando a energia potencial for menor que 7J, o sistema estará em uma região estável. 
# O movimento será de oscilação em torno da posição de equilíbrio. 
# Quando a energia potencial for maior que 8J, o sistema estará em uma região instável.
# O movimento será de uma oscilação amplificada. 

# alínea b)

x0 = 2.2            #posiçao inicial
v0 = 0              #velocidade inicial
ti = 0              #tempo inicial
tf = 20             #tempo final
dt = 0.001
n = int((tf-ti)/dt)

t = np.linspace(ti, tf, n+1)
def oscCubico(x0, v0, k, m, n, dt, alpha):          # função que calcula a posição, velocidade, aceleração e 
                                                    # energia mecânica, num oscilador cúbico
    #inicializaçao dos arrays
    x = np.empty(n+1)
    v = np.empty(n+1)
    a = np.empty(n+1)
    Em = np.empty(n+1)
    x[0] = x0
    v[0] = v0

    for i in range(n):
        a[i] = (-k/m*x[i]) - (3/m*alpha)*(x[i]**2)
        v[i+1] = v[i] + a[i] * dt
        x[i+1] = x[i] + v[i+1] * dt
        Em[i] = 0.5*m*(v[i]**2) + (0.5*k*(x[i]**2) + alpha*(x[i]**3))

    Em[n] = 0.5*m*(v[n]**2) + (0.5*k*(x[n]**2) + alpha*(x[n]**3))

    return x, v, a, Em


valores = oscCubico(x0,v0,k,m,n,dt,alpha)
Em = valores[3]
print("A energia mecânica é de", Em[n], "J")
plt.plot(t,Em)
plt.grid()
plt.show()

# Resposta:
# A energia mecânica é de 2.95 J

# alínea c)

ti_estac = 0

def fourier(x0,v0,m,k,alpha,t,n,ti_estac, dt):          # função que calcula os coeficientes de fourier
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
        a[i]=(-k/m*x[i]) - ((3/m*alpha)*(x[i]**2))
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

def fourier_calc(t,x,t0,t1,dt,nf):              # análise de Feurier
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

fourier(x0,v0,m,k,alpha,t,n,ti_estac, dt)