#========================================================================================
#==========================PROJETO INTEGRADO TERCEIRO TRIMESTRE==========================
#===================================JULIANO & LINDSEY====================================
#=============================PROFESSORES ANA LETICIA & ALAN============================= 
#===============================TURMA 31TE - ETEFMC - 2022===============================
#========================================================================================

import sys
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
Fs = 150000 # Frequencia de amostragem
t = np.linspace(0,1,Fs) # Eixo X do sinal
x1 = np.sin(2 * np.pi * 4400 * t)# 4400Hz
x2 = np.sin(2 * np.pi * 900 * t) # 900Hz
x3 = np.sin(2 * np.pi * 2900 * t) # 2900Hz
x = x1 + x2 + x3 # Somando os sinais

# Plotando o sinal
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, x1)
ax1.set_title('Sinal 1')
ax2.plot(t, x)
ax2.set_title('Soma Sinais')

np.savetxt('sinalEntrada.txt', x, delimiter=' ', fmt='%1.4e') # Salvando o sinal X no .txt

plt.tight_layout()
plt.show()
