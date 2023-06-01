import matplotlib.pyplot as plt
import numpy as np

#bola de ténis batida junta ao solo 
#angulo de 45º

#Se não houver resistência do ar, não existem forças não 
# conservativas a atuar no sistema, o que implica que a energia 
# mecânica será constante ao longo do tempo, logo podemos calcular 
# a energia no instante inicial onde Em=Ec
# pois como a bola está ao nível do chão a energia potencial 
# gravítica é nula.

# Parâmetros
dt = 0.0001
t0 = 0
tf = 0.8
g = 9.8 # Aceleração gravítica

m = 0.057 # Massa do corpo
x0 = np.array([0, 0]) # Posição inicial

# Velocidade inicial
angle = np.radians(10)
v0Norm = 250/9
v0 = np.array([
    v0Norm * np.cos(angle),
    v0Norm * np.sin(angle)
])

# Constantes para a resistência do ar
vT = 250/9 # velocidade terminal
D = g/vT**2

# Esta função calcula a força da resistência do ar a partir da velocidade atual
def forcaRes(v):
    vNorm = np.linalg.norm(v)
    vHat = v / vNorm
    return -m * D * vNorm** 2 * vHat

# Esta função calcula a aceleração a partir da velocidade atual
def accel(v):
    return np.array([0, -g]) + forcaRes(v)/m

def energiaMecanica(x, v):
    # Energia cinética
    E_c = 1/2 * m * np.linalg.norm(v)**2
    # Energia potencial
    E_p = m * g * x[1]
    # Energia mecânica
    return E_c + E_p
    

# Número de passos/iterações
#
# + 0.1 para garantir que não há arrendodamentos
# para baixo
n = int((tf-t0) / dt + 0.1)
# Movimento 2D
shape = (n + 1, 2)

a = np.zeros(shape)
v = np.zeros(shape)
x = np.zeros(shape)
t = np.zeros(n + 1)

E_m = np.zeros(n + 1)

# Valores iniciais
a[0] = accel(v0)
v[0] = v0
x[0] = x0
t[0] = t0

E_m[0] = energiaMecanica(x0, v0)

for i in range(n):
    a[i + 1] = accel(v[i])
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i] * dt
    t[i + 1] = t[i] + dt
    
    E_m[i + 1] = energiaMecanica(x[i + 1], v[i + 1])

plt.plot(t, E_m, "g")
plt.xlabel("t (s)")
plt.ylabel("Energia (J)")
plt.title("Energia mecânica")
plt.show()




