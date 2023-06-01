import numpy as np
import matplotlib.pyplot as plt

#bola de futebol chutada

v0 = 100    #velocidade inicial = 100km/h
a = 10      #angulo com a horizontal
g = 9.8     #gravidade
dt = 0.01   #tempo do passo
t0 = 0      #tempo inicial
n = 1000    #numero de passos

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
    vy[i + 1] = vy[i] - g * dt

#plotar a trajetoria
plt.plot(px, py)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajetoria da bola de futebol")
plt.show()




