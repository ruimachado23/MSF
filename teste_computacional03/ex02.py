# Rui Machado 113765
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
m = 1.0
k = 1.0
b = 0.02
alpha = 0.15
F0 = 7.5
wf = 1.0

# alínea a)

# Função que define as equações diferenciais do sistema
def equations_of_motion(t, y):
    x, v = y
    dx_dt = v
    dv_dt = (F0 * np.cos(wf * t) - b * v - 4 * alpha * (x**3)) / m
    return [dx_dt, dv_dt]

# Método de Runge-Kutta de 4ª ordem
def runge_kutte4(t, y, h):
    k1 = h * np.array(equations_of_motion(t, y))
    k2 = h * np.array(equations_of_motion(t + h/2, y + k1/2))
    k3 = h * np.array(equations_of_motion(t + h/2, y + k2/2))
    k4 = h * np.array(equations_of_motion(t + h, y + k3))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Condições iniciais
t0 = 0.0
y0 = [2.0, 0.0]

# Configuração do tempo
dt = 0.01
t_max = 20.0
t = np.arange(t0, t_max, dt)

# Solução numérica usando o método de Runge-Kutta de 4ª ordem
x_list = []
for ti in t:
    x_list.append(y0[0])
    y0 = runge_kutte4(ti, y0, dt)

# Plot da posição (x) em função do tempo
plt.plot(t, x_list)
plt.xlabel('Tempo')
plt.ylabel('Posição (x)')
plt.grid()
plt.show()
