#113765 Rui Machado

#import de bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import math

#parametros
g = 9.80                                

deltat = 0.001
t0 = 0
tf = 10

angulo = np.radians(0)                 #graus

x0 = 0                                 #metros
vx0 = 30                               #metros por segundo
ax0 = 0                                #metros por segundo ao quadrado
rotacao_x = 0                          #radaianos por segundo
y0 = 3
vy0 = 0
ay0 = -g
rotacao_y = 0
z0 = 0
vz0 = 0
az0 = 0
rotacao_z = -60

vterminal = 20          #metros por segundo

massa = 0.057           #quilogramas
raio = 0.034            #metros
area = math.pi*raio**2  #metros quadrados

pressao = 1.225         #quilogramas por metro ao cubo

dres = g/vterminal**2   #coeficiente de arrasto ou coeficiente de resistência do ar

#inicialização
n = np.int_((tf-t0)/deltat)

t = np.zeros(n+1)            
t[0] = t0

x = np.zeros(n+1)
vx = np.zeros(n+1)
ax = np.zeros(n+1)
x[0] = x0
vx[0] = vx0
ax[0] = ax0

y = np.zeros(n+1)
vy = np.zeros(n+1)
ay = np.zeros(n+1)
y[0] = y0
vy[0] = vy0
ay[0] = ay0

z = np.zeros(n+1)
vz = np.zeros(n+1)
az = np.zeros(n+1)
z[0] = z0
vz[0] = vz0
az[0] = az0

F_magnus_x = np.zeros(n+1)
F_magnus_y = np.zeros(n+1)
F_magnus_z = np.zeros(n+1)


#for loops das grandezas

i = 0

while y[i] >= 0:
    #atualização do tempo
    t[i+1] = t[i] + deltat

    #atualização do modulo da velocidade
    modulo_velocidade = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)

    #atualização da força de magnus
    F_magnus_x[i] = (0.5 * area* pressao * raio * (rotacao_y * vz[i] - rotacao_z * vy[i]))/massa
    F_magnus_y[i] = (0.5 * area* pressao * raio * (rotacao_z * vx[i] - rotacao_x * vz[i]))/massa
    F_magnus_z[i] = (0.5 * area* pressao * raio * (rotacao_x * vy[i] - rotacao_y * vx[i]))/massa

    #atualização da posição
    x[i+1] = x[i] + vx[i]*deltat + 1/2 * ax[i]*deltat**2
    y[i+1] = y[i] + vy[i]*deltat + 1/2 * ay[i]*deltat**2
    z[i+1] = z[i] + vz[i]*deltat + 1/2 * az[i]*deltat**2

    #atualização da velocidade
    vx[i+1] = vx[i] + ax[i]*deltat
    vy[i+1] = vy[i] + ay[i]*deltat
    vz[i+1] = vz[i] + az[i]*deltat

    #atualização da aceleração
    ax[i+1] = - dres * vx[i] * modulo_velocidade + F_magnus_x[i]
    ay[i+1] = - dres * vy[i] * modulo_velocidade + F_magnus_y[i] - g
    az[i+1] = - dres * vz[i] * modulo_velocidade + F_magnus_z[i]    

    #verificar quando atinge um valor
    if ((12 - 0.01) < x[i] < (12 + 0.01)) and (y[i] > 1):
        print("O serviço passou a rede") 

    i += 1

valido = False
for i in range(n):
    if ((0 - 0.01) < y[i] < (0 + 0.01)) and (12 < x[i] <= 18.4) :
        print("O serviço foi válido!")
        valido = True
        break
if not valido:
    print("Falhou o Serviço!")

for i in range(n):
    if y[i]*y[i+1] < 0:
        alcance = x[i]

#output
print('Distancia que a bola cai no solo: ', alcance)

#resultado:
#O serviço passou a rede
#O serviço foi válido!
#Distancia que a bola cai no solo:  18.131203401321027
