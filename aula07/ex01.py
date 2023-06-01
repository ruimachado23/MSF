import numpy as np
import matplotlib.pyplot as plt

#bola de ténis raquetiada
#sem rotacao
#3 dimensões

#Dados
p0 = (-10,1,0)      #posicao inicial
v0 = 130            #velocidade inicial 
a = 10              #angulo com a horizontal
m = 0.57            #massa da bola
r = (67/2) * 10**-6 #raio da bola
vt = 100            #velocidade terminal
g = 9.8             #gravidade
dt = 0.01           #tempo do passo
t0 = 0              #tempo inicial
n = 1000            #numero de passos
A = np.pi * r**2    #area da bola
d = g / vt**2       #coeficiente de resistencia do ar

#funcão para calcular a força de resistencia do ar
def forcaResAr(D, v, m):
    v_norm = np.linalg.norm(v)
    v_hat = v / v_norm
    return -m * D * v_norm**2 * v_hat

#aceleracao com resistencia do ar
a = g - forcaResAr(d, v0, m) 

#inicializar arrays para a posicao
px = np.empty(n + 1)
px[0] = 0

py = np.empty(n + 1)
py[0] = 0

#inicializar arrays para a velocidade
vx = np.empty(n + 1)
vx[0] = v0 * np.cos(a)

vy = np.empty(n + 1)
vy[0] = v0 * np.sin(a)

#for loop para calcular as posicoes e as velocidades
for i in range(n):
    #calcular a posicao
    px[i + 1] = px[i] + vx[i] * dt
    py[i + 1] = py[i] + vy[i] * dt

    #calcular a velocidade
    vx[i + 1] = vx[i]
    vy[i + 1] = vy[i] - a * dt

#plotar a trajetoria
plt.ylim(0, 130)
plt.plot(px, py)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajetoria da bola de futebol com resistencia do ar")  

#distancia maxima 
dmax = np.max(px)
print("A distancia maxima atingida é: ", dmax)

#gŕafico do tempo de voo
t = np.linspace(t0, n*dt, n+1)
plt.figure()
plt.ylim(0, 80)
plt.plot(t, py)
plt.xlabel("t (s)")
plt.ylabel("y (m)")
plt.title("Altura da bola de futebol com resistencia do ar")

#tempo de voo
tmax = np.max(t)
print("O tempo de voo é: ", tmax)

plt.show()






