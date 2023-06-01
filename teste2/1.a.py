#113765 Rui Machado

#Determinar a distância a que bola cai do solo
#Determinar a validez do serviço

#import de bibliotecas
import numpy as np
import matplotlib.pyplot as plt

g = 9.8
vi = 30                     #velocidade inicial (m/s)
ang = np.radians(0)         #ângulo de lançamento (rad)
vx0 = vi * np.cos(ang)
vy0 = vi * np.sin(ang)
v0 = [vx0, vy0, 0]          #velocidade inicial da bola (m/s)s
r0 = [0, 3, 0]              #posição inicial da bola (m)
a0 = [0, -g, 0]             #aceleração da bola (m/s^2)
r = 0.034                   #raio da bola (m)
m = 0.057                   #massa da bola (kg)
vT = 20 * 1000 / 3600       #velocidade terminal (m/s)
dar = 1.225                 #densidade do ar (kg/m^3)
w = [0, 0, 0]               #velocidade angular da bola (rad/s)

ti = 0
tf = 10
dt = 0.001
n = int((tf - ti) / dt)

t = np.linspace(ti, tf, n + 1)


def prodExt(a, b):
    return (a[1] * b[2] - b[1] * a[2], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def magnus_3D(r0, v0, a0, rot, p_ar, r, n, dt, vt, m):
    g = 9.80
    A = np.pi * r ** 2
    apr = 0.5 * p_ar * A * r

    x = np.empty(n + 1)
    y = np.empty(n + 1)
    z = np.empty(n + 1)

    vx = np.empty(n + 1)
    vy = np.empty(n + 1)
    vz = np.empty(n + 1)

    ax = np.empty(n + 1)
    ay = np.empty(n + 1)
    az = np.empty(n + 1)

    x[0] = r0[0]
    y[0] = r0[1]
    z[0] = r0[2]

    vx[0] = v0[0]
    vy[0] = v0[1]
    vz[0] = v0[2]

    ax[0] = a0[0]
    ay[0] = a0[1]
    az[0] = a0[2]

    dres = g / vt ** 2
    for i in range(n):
        vv = np.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2)
        rot_v = prodExt(rot, (vx[i], vy[i], vz[i]))

        mag_x = apr * rot_v[0] / m
        mag_y = apr * rot_v[1] / m
        mag_z = apr * rot_v[2] / m

        ax[i] = a0[0] - dres * vv * vx[i] + mag_x
        ay[i] = a0[1] - dres * vv * vy[i] + mag_y
        az[i] = a0[2] - dres * vv * vz[i] + mag_z

        vx[i + 1] = vx[i] + ax[i] * dt
        vy[i + 1] = vy[i] + ay[i] * dt
        vz[i + 1] = vz[i] + az[i] * dt

        x[i + 1] = x[i] + vx[i] * dt
        y[i + 1] = y[i] + vy[i] * dt
        z[i + 1] = z[i] + vz[i] * dt

        if x[i] < 20 + dt and x[i + 1] > 20 - dt:
            print("Altura quando atinge 20m = ", y[i])

    return (x, y, z), (vx, vy, vz), (ax, ay, az)

values = magnus_3D(r0, v0, a0, w, dar, r, n, dt, vT, m)
x = values[0][0]
y = values[0][1]
z = values[0][2]

valido = False

for i in range(n):
    if ((0 - 0.01) < y[i] < (0 + 0.01)) and (12 < x[i] < 18):
        print("O serviço é válido!")
        print("Alcance:", x[i], "m")
        valido = True
        break

if not valido:
    print("O serviço é inválido!")

for i in range(n):
    if y[i]*y[i+1] < 0:
        alcance = x[i]
        
print('A bola cai no solo a uma distância igual a', alcance, "metros")

plt.plot(x, y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajetória da bola")
plt.ylim(0, 4)

# Plotar da rede
plt.axhline(y=1, color='r', linestyle='--', label='Rede')

plt.show()

#resultado:
#O serviço é inválido!
#A bola cai no solo a uma distância igual a 7.341976791374359 metros