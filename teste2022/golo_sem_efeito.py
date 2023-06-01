import matplotlib.pyplot as plt
import numpy as np

#jogador de futebol remata a bola
#sem efeito
#3 dimensões

d = 20                              #distância da baliza
v0 = 100                            #velocidade inicial da bola em km/h
a = 16                              #ângulo com a horizontal em graus
r = 0.11                            #raio da bola em m
m = 0.45                            #massa da bola em kg
PAr = 1.225                         #densidade do ar em kg/m^3  
dt = 0.00001 
t = np.arange(0, 0.5+dt, dt)
t0 = 0                              #tempo inicial em s
A = np.pi*(r)**2                    #area da bola em m^2
g = 9.8
vterminal = 100/3.6
D = g/(vterminal**2)                #coeficiente para resistencia do ar

Rx = np.zeros(t.size)
Ry = np.zeros(t.size)
Rz = np.zeros(t.size)

Vx = np.zeros(t.size)
Vy = np.zeros(t.size)
Vz = np.zeros(t.size)

#Posição inicial (0,0,0)

#Velocidade inicial --> 100km/h
Vx[0] = 100/3.6 * np.cos(a*np.pi/180)
Vy[0] = 100/3.6 * np.sin(a*np.pi/180)
Vz[0] = 0

for i in range(0,t.size-1):
    v = np.sqrt(Vx[i]**2+Vy[i]**2+Vz[i]**2) #modulo da velocidade
    
    ax = -D * Vx[i] * abs(v) 
    ay = -g - D * Vy[i] * abs(v)
    az = -D * Vz[i] * abs(v) 
    
    Vx[i+1] = Vx[i] + ax * dt
    Vy[i+1] = Vy[i] + ay * dt
    Vz[i+1] = Vz[i] + az * dt
    
    Rx[i+1] = Rx[i] + Vx[i] *dt
    Ry[i+1] = Ry[i] + Vy[i] * dt
    Rz[i+1] = Rz[i] + Vz[i] * dt

    if Rx[i] > 20 and 0 < Ry[i] < 2.4 and -3.75 < Rz[i] < 3.75:
        print("É Golo!")
    
    else:
        print("Falhou, não é golo")

plt.plot(t, Rx, label="x(t)")
plt.plot(t, Ry, label="y(t)")
plt.plot(t, Rz, label="z(t)")
plt.legend()
plt.grid()
plt.show()

