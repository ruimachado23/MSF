import matplotlib.pyplot as plt
import numpy as np

#pessoa a viajar de trotinete elétrica
#objetivo --> Quanto tempo leva a percorrer 2km?

m = 60 + 12          #massa do corpo
p = 0.48 * 735.4975  #potência do corpo
rho_ar = 1.225       #densidade do ar
A = 0.50             #área frontal do corpo
C_res = 0.9          #coeficiente de resistência do ar
mu = 0.01            #coeficiente de resistência do alcatrão
g = 9.8              #aceleração gravítica na terra

dt = 0.001
t0 = 0
tf = 500
x0 = 0
v0 = 0.5

#calcula a aceleracao
def accel(v):
    #aceleracao pela potencia do corpo
    accel_p = p/(m * v)
    #aceleracao pela resistencia do ar
    accel_res = -C_res/(2*m) * A * rho_ar * v**2

    #aceleracao pelo atrito
    accel_atrito = - mu * g

    #aceleracao pelo peso ----> nao considerar, não está em queda livre

    #aceleracao total
    return accel_p + accel_res + accel_atrito 

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

for i in range(n):
  # Subtrair 2km a posição
  scaledX0 = x[i] - 2000
  scaledX1 = x[i + 1] - 2000
  # Procurar os zeros com a posição modificada
  if scaledX0 == 0 or scaledX0 * scaledX1 < 0:
    idx = i
    break

x2000 = x[idx]
t2000 = t[idx]

print("Posição: {:.2f} m".format(x2000))
print("Tempo: {:.2f} s".format(t2000))

plt.plot(t, x - 2000, "r")
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("Posição")
plt.show()