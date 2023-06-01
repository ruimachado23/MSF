import matplotlib.pyplot as plt
import numpy as np

#ciclista
g = 9.8             #aceleração gravítica na terra
mu = 0.004          #coeficiente de resistência do alcatrão
rho_ar = 1.225      #densidade do ar
A = 0.3             #área frontal do ciclista-bicicleta
m = 75              #massa do ciclista-bicicleta
C_res = 0.9         #coeficiente de resistência do ar

#a unidade SI de potência é o watt, logo precisamos de converter de cavalos para watts
p_ciclista = 0.4 * 735.4987

#parâmetros
dt = 0.001
t0 = 0
tf = 200
x0 = 0
v0 = 1

# esta função calcula a aceleração a partir da velocidade atual do ciclista
def accel(v):
    # Aceleração pela potência do ciclista
    accel_p = p_ciclista/(m * v)
    # Aceleração pela resistência do ar
    accel_res = -C_res/(2*m) * A * rho_ar * v**2
    # Aceleração pelo atrito
    accel_atrito = - mu * g
    # Aceleração total
    return accel_p + accel_res + accel_atrito

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

idx = v.argmax()
vT = v[idx]

for i in range(n):
  # Subtrair 90% da velocidade terminal a velocidade
  scaledV0 = v[i] - vT * 0.9
  scaledV1 = v[i + 1] - vT * 0.9
  # Procurar os zeros com a velocidade modificada
  if scaledV0 == 0 or scaledV0 * scaledV1 < 0:
    idx = i
    break

v90 = v[idx]
t90 = t[idx]

print("Velocidade a 90% da velocidade terminal: {:.2f} m/s".format(v90))
print("Tempo a 90% da velocidade terminal: {:.2f} s".format(t90))

plt.plot(t, v - 0.9 * vT, "g")
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.title("Velocidade (- 90% Velocidade Terminal)")
plt.show()