import matplotlib.pyplot as plt
import numpy as np

#subida com inclinação

g = 9.8             #aceleração gravítica na terra
mu = 0.004          #coeficiente de resistência do alcatrão
rho_ar = 1.225      #densidade do ar
A = 0.3             #área frontal do ciclista-bicicleta
m = 75              #massa do ciclista-bicicleta
C_res = 0.9         #coeficiente de resistência do ar

#a unidade SI de potência é o watt, logo precisamos de converter de cavalos para watts
p_ciclista = 0.4 * 735.4987

# Parâmetros
dt = 0.001
t0 = 0
tf = 500
x0 = 0
v0 = 1

# Inclinação em radianos
incl = np.radians(5)

# Esta função calcula a aceleração a partir da velocidade atual do ciclista
def accel(v):
    # Aceleração pela potência do ciclista
    accel_p = p_ciclista/(m * v)
    # Aceleração pela resistência do ar
    accel_res = -C_res/(2*m) * A * rho_ar * v**2
    # Aceleração pelo atrito
    accel_atrito = - mu * np.cos(incl) * g
    # Aceleração pelo peso
    accel_peso = - np.sin(incl) * g
    # Aceleração total
    return accel_p + accel_res + accel_atrito + accel_peso

# Número de passos/iterações
#
# + 0.1 para garantir que não há arrendodamentos
# para baixo
n = int((tf-t0) / dt + 0.1)

t = np.zeros(n + 1)
x = np.zeros(n + 1)
v = np.zeros(n + 1)
a = np.zeros(n + 1)

# Valores iniciais
a[0] = accel(v0)
v[0] = v0
x[0] = x0
t[0] = t0

for i in range(n):
    a[i + 1] = accel(v[i])
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i] * dt
    t[i + 1] = t[i] + dt

v = (30 * 1000) / 3600

forca_res = C_res/2 * A * rho_ar * v**2
forca_atrito = + mu * g * m

potencia = (forca_res + forca_atrito) * v

print("A potencia é " + potencia)

