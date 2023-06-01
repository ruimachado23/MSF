#113765 Rui Machado

#calcular o trabalho feito pelo motor durante a viagem

#import de bibliotecas
import math

# Dados do problema
massa = 2000                                    #kg
coeficiente_resistencia_rolamento = 0.04
coeficiente_resistencia_ar = 0.25
area_frontal = 2                                #m^2
densidade_ar = 1.225                            #kg/m^3
potencia = 40 * 10**3                           #kW para W
velocidade_inicial = 1                          #m/s
inclinacao = 5                                  #graus

# Conversão da inclinação para radianos
inclinacao_rad = math.radians(inclinacao)

# Definição da aceleração da gravidade
aceleracao_gravidade = 9.8  # m/s^2

# Inicialização das listas para armazenar os valores ao longo do tempo
posicao = [0]  # posição inicial é 0
velocidade = [velocidade_inicial]  

# Intervalo de tempo (∆t)
delta_t = 0.1  

# Iteração para calcular a evolução temporal
tempo = 0  

trabalho_total = 0  

while posicao[-1] < 2000:  
    # Cálculo das forças
    Ftr = potencia / velocidade[-1]
    Far = 0.5 * coeficiente_resistencia_ar * area_frontal * densidade_ar * velocidade[-1]**2
    Fr = coeficiente_resistencia_rolamento * massa * aceleracao_gravidade * math.cos(inclinacao_rad)
    
    # Cálculo da força resultante
    Fresultante = Ftr - Far - Fr
    
    # Cálculo da aceleração
    aceleracao = Fresultante / massa
    
    # Cálculo da velocidade usando o método de Euler
    nova_velocidade = velocidade[-1] + aceleracao * delta_t
    
    # Cálculo da posição usando o método de Euler
    nova_posicao = posicao[-1] + velocidade[-1] * delta_t
    
    # Cálculo do trabalho realizado pelo motor
    trabalho_motor = potencia * delta_t
    
    # Atualização do trabalho total
    trabalho_total += trabalho_motor
    
    # Atualização das listas
    velocidade.append(nova_velocidade)
    posicao.append(nova_posicao)
    
    # Atualização do tempo
    tempo += delta_t

#output
print(f"Trabalho feito pelo motor durante a viagem: {trabalho_total:.1f} J")

#resultado:
#Trabalho feito pelo motor durante a viagem: 3100000.0 J