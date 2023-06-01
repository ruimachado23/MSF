import numpy as np
import matplotlib.pyplot as plt
import math

# m, b, r2 = np.polyfit(x, y, deg) (para deg = 1)
# logx = np.log(x)

# Densidade do ar
PAr = 1.225

class ChartsPlotter():
    def __init__(self):
        self.subplots = []

    def addCharts(self, x, *args, title=None):
        self.subplots.append((x, args, title))

    def plotAll(self):
        subplotsN = len(self.subplots)

        rows = int(math.floor(math.sqrt(subplotsN)+0.01))
        cols = int(math.ceil(subplotsN / rows))

        i = 1
        for (x, Ys, title) in self.subplots:
            plt.subplot(rows, cols, i)
            i+=1
            for y in Ys:
                plt.plot(x, y)
            if title is not None:
                plt.title(title)
        
        plt.show()

def regressao_linear(x, y):
    if len(x) != len(y):
        raise "Tamanho dos arrays tem de ser iguais"
    N = len(x)

    produtos = []
    for i in range(N):
        produtos.append(x[i] * y[i])

    quadrados = []
    for i in range(N):
        quadrados.append(x[i] ** 2)

    quadradosY = []
    for i in range(N):
        quadradosY.append(y[i] ** 2)

    m = (N * sum(produtos) - sum(x) * sum(y)) / (N * sum(quadrados) - sum(x) ** 2)

    b = (sum(quadrados) * sum(y) - sum(x) * sum(produtos)) / (N * sum(quadrados) - sum(x) ** 2)

    r2 = ((N * sum(produtos) - sum(x) * sum (y)) ** 2) / ((N * sum(quadrados) - sum(x) ** 2) * (N * sum(quadradosY) - sum(y) ** 2))

    deltam = abs(m) * math.sqrt((1/r2 - 1) / (N - 2))
    deltab = deltam * math.sqrt(sum(quadrados) / N)

    return m, deltam, b, deltab, r2


def to_ms(x):
    return x / 3.6

def to_kmh(x):
    return x * 3.6

def aproximacao_retangular(function, inicial, final, dt=0.01):
	f = function(np.arange(inicial, final + dt, dt))
	return dt * (np.sum(f))

def aproximacao_trapezoidal(function, inicial, final, dt=0.01):
	f = function(np.arange(inicial, final + dt, dt))
	return dt * ((f[0] + f[-1]) * 0.5 + np.sum(f[1:-1]))


def norma(arr):
	return math.sqrt(sum([el**2 for el in arr]))

def a_resistencia_ar(v, g, vt):
    D = g / (vt**2)
    return - D * norma(v) * v

def a_magnus(v, w, r, m):
        vw = np.cross(v, w)
        magnus = 1/2 * math.pi*(r**3) * PAr * vw
        return -magnus/m

"""""""""""""""""""""""""""""""""""""""""
"										"
"	Euler(-cromer) Method				"
"										"
"""""""""""""""""""""""""""""""""""""""""

def queda_livre(h, g_vector, tf=1, dt=0.001):
    return euler(np.array([h]), np.zeros(1), lambda t, r, v: g_vector, tf, dt)

def queda_resistencia_ar(vt, h, g_vector, tf=1, dt=0.001):
	return euler(np.array([h]), np.zeros(1), lambda t, r, v: g_vector + a_resistencia_ar(v, norma(g_vector), vt), tf, dt)
	

def queda_fmagnus_resistencia_ar(vt, h, w, r, m, g_vector, tf=1, dt=0.001):
	return euler(np.array([h]), np.zeros(1), lambda t, r, v: g_vector + a_resistencia_ar(v, norma(g_vector), vt) + a_magnus(v, w, r, m), tf, dt)

def projetil_livre(r0, v0, g_vector, tf=1, dt=0.001):
    return euler(r0, v0, lambda t, r, v: g_vector, tf, dt)

def projetil_resistencia_ar(r0, v0, vt, g_vector, tf=1, dt=0.001):
	return euler(r0, v0, lambda t, r, v: g_vector + a_resistencia_ar(v, norma(g_vector), vt), tf, dt)

def projetil_fmagnus_resistencia_ar(r0, v0, vt, w, r, m, g_vector, tf=1, dt=0.001):
	return euler(r0, v0, lambda t, r, v: g_vector + a_resistencia_ar(v, norma(g_vector), vt) + a_magnus(v, w, r, m), tf, dt)

def euler(r0, v0, atv, tf=1, dt=0.001, cromer=True):
    if len(r0) != len(v0):
        raise "Tamanho dos arrays tem de ser iguais"
    
    dim = len(r0)

    n = int(tf/dt)

    t = np.arange(0, tf+dt, dt)
    r = np.zeros([n+1, dim])
    v = np.zeros([n+1, dim])
    a = np.zeros([n+1, dim])

    t[0] = 0
    r[0] = r0
    v[0] = v0

    for i in range(n):
        t[i+1] = t[i] + dt

        if i == 0:
            a[0] = atv(t[i+1], r[i], v[i])
        a[i+1] = atv(t[i+1], r[i], v[i])

        for j in range(dim):
            v[i+1, j] = v[i, j] + a[i, j] * dt
            r[i+1, j] = r[i, j] + v[(i+1) if cromer else i, j] * dt

    return t, r, v, a


def get_axis(arr, axis):
	return arr[:, axis]

def find_index(arr, time):
    for i in range(len(arr)):
        if arr[i] >= time:
            return i
    return i