import numpy as np
m= 1
g = 9.8             #aceleração gravítica na terra
mu = 0.004          #coeficiente de resistência do alcatrão
rho_ar = 1.225      #densidade do ar
A = 0.3             #área frontal do ciclista-bicicleta
C_res = 0.9         #coeficiente de resistência do ar
incl = np.radians(5)
p = 0.4 * 735.4987

# Esta função calcula a aceleração a partir da velocidade atual do ciclista
def accel(v):
    # Aceleração pela potência do ciclista
    accel_p = p/(m * v)
    # Aceleração pela resistência do ar
    accel_res = -C_res/(2*m) * A * rho_ar * v**2
    # Aceleração pelo atrito
    accel_atrito = - mu * np.cos(incl) * g
    # Aceleração pelo peso
    accel_peso = - np.sin(incl) * g
    # Aceleração total
    return accel_p + accel_res + accel_atrito + accel_peso