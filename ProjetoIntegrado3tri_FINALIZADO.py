#========================================================================================
#==========================PROJETO INTEGRADO TERCEIRO TRIMESTRE==========================
#===================================JULIANO & LINDSEY====================================
#=============================PROFESSORES ANA LETICIA & ALAN============================= 
#===============================TURMA 31TE - ETEFMC - 2022===============================
#========================================================================================

# IMPORTANDO AS BIBLIOTECAS
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys
from scipy.signal import sosfilt, sosfilt_zi
import json

# FUNCOES DE GRAFICO DE MAGNITUDE
def mfreqzOS(zOS, pOS, kOS, Fs): # Filtro Lowpass do One-Seg
    wzOS, hzOS = signal.freqz_zpk(zOS, pOS, kOS)
    hzOS[hzOS == 0.00000000e+00+0.00000000e+00j] = 0.00000001e+00+0.00000001e+00j
    MagOS = 20*np.log10(abs(hzOS))
    FreqOS = wzOS*Fs/(2*np.pi)
    figOS = plt.figure(figsize=(10, 6))
 
    sub1 = plt.subplot(1, 1, 1)
    sub1.plot(FreqOS, MagOS, 'r', linewidth=2)
    sub1.axis([1, Fs/10, -100, 5])
    sub1.set_title('One-Seg', fontsize=20)
    sub1.set_xlabel('Frequência [Hz]', fontsize=20)
    sub1.set_ylabel('Amplitude [dB]', fontsize=20)
    sub1.grid()
 
    plt.subplots_adjust(hspace=0.5)
    figOS.tight_layout()
    plt.show()

def mfreqzSD(zSD, pSD, kSD, Fs): # Filtro Bandpass do SD
    wzSD, hzSD = signal.freqz_zpk(zSD, pSD, kSD)
    hzSD[hzSD == 0.00000000e+00+0.00000000e+00j] = 0.00000001e+00+0.00000001e+00j
    MagSD = 20*np.log10(abs(hzSD))
    PhaseSD = np.unwrap(np.arctan2(np.imag(hzSD), np.real(hzSD)))*(180/np.pi)
    FreqSD = wzSD*Fs/(2*np.pi)
    figSD = plt.figure(figsize=(10, 6))

    sub1 = plt.subplot(1, 1, 1)
    sub1.plot(FreqSD, MagSD, 'r', linewidth=2)
    sub1.axis([1, Fs/10, -100, 5])
    sub1.set_title('Standart Definition (SD)', fontsize=20)
    sub1.set_xlabel('Frequência [Hz]', fontsize=20)
    sub1.set_ylabel('Amplitude [dB]', fontsize=20)
    sub1.grid()

    plt.subplots_adjust(hspace=0.5)
    figSD.tight_layout()
    plt.show()

def mfreqzHD(zHD, pHD, kHD, Fs): # Filtro Highpass do HD
    wzHD, hzHD = signal.freqz_zpk(zHD, pHD, kHD)
    hzHD[hzHD == 0.00000000e+00+0.00000000e+00j] = 0.00000001e+00+0.00000001e+00j
    MagHD = 20*np.log10(abs(hzHD))
    FreqHD = wzHD*Fs/(2*np.pi)
    figHD = plt.figure(figsize=(10, 6))
    
    sub1 = plt.subplot(1, 1, 1)
    sub1.plot(FreqHD, MagHD, 'r', linewidth=2)
    sub1.axis([1, Fs/10, -100, 5])
    sub1.set_title('High Definition (HD)', fontsize=20)
    sub1.set_xlabel('Frequência [Hz]', fontsize=20)
    sub1.set_ylabel('Amplitude [dB]', fontsize=20)
    sub1.grid()
 
    plt.subplots_adjust(hspace=0.5)
    figHD.tight_layout()
    plt.show()

np.set_printoptions(threshold=sys.maxsize) #Definindo como máximo os valores dos arrays
x = np.loadtxt('sinalEntrada.txt') # Arquivo do SINAL de entrada
saidaOS = "saidaOS.json"
saidaSD = "saidaSD.json"
saidaHD = "saidaHD.json"


# PARAMETROS DOS FILTROS
with open('parametrosEntrada.json') as f:
    parametros = json.load(f)
    Fs = parametros["Fs"]
    wmOS = parametros["wmOS"]
    wmSD = parametros["wmSD"]
    wmHD = parametros["wmHD"]

Fs = int(Fs) # Frequencia de amostragem
wmOS = int(wmOS) # Frequencia do sinal One-Seg
wmSD = int(wmSD) # Frequencia do sinal SD
wmHD = int(wmHD) # Frequencia do sinal HD

Ap = 1 # Ondulacao da banda passante [dB]
As = 40 # Atenuacao da banda de rejeicao [dB]


wpOS = (wmOS + 100)/(Fs/2) # Banda passante One-seg (Normalizada)
wsOS = (wmOS + 600)/(Fs/2) # Banda de rejeicao One-seg (Normalizada)

fpSD = np.array([(wmSD - 350), (wmSD + 350)]) # Banda passante SD
fsSD = np.array([(wmSD - 900), (wmSD + 900)]) # Banda de rejeicao SD
wpSD = fpSD/(Fs/2) # Normalizando
wsSD = fsSD/(Fs/2) # Normalizando

wsHD = (wmHD - 1000)/(Fs/2) # Banda passante HD (Normalizada)     
wpHD = (wmHD - 500)/(Fs/2) # Banda de rejeicao HD (Normalizada)


# COMPUTANDO A ORDEM DOS FILTROS
NSD, wcSD = signal.buttord(wpSD, wsSD, Ap, As, analog=True)
NHD, wcHD = signal.buttord(wpHD, wsHD, Ap, As)
NOS, wcOS = signal.buttord(wpOS, wsOS, Ap, As)

print('Ordem do filtro LowPass:', NOS)
print('Ordem do filtro BandPass:', NSD)
print('Ordem do filtro HighPass:', NHD)


# PROJETANDO OS FILTROS
sosSD = signal.butter(NSD, wcSD, 'bandpass', False, 'sos')
zSD, pSD, kSD = signal.butter(NSD, wcSD, 'bandpass', False, 'zpk')


sosOS = signal.butter(NOS, wcOS, 'lowpass', False, 'sos')
zOS, pOS, kOS = signal.butter(NOS, wcOS, 'lowpass', False, 'zpk')

zHD, pHD, kHD = signal.butter(NHD, wcHD, 'highpass', False, 'zpk')
sosHD = signal.butter(NHD, wcHD, 'highpass', False, 'sos')


# FILTRANDO
t = np.linspace(0,1,Fs) # Eixo X do gráfico

# Filtro LOWPASS
yOS = sosfilt(sosOS,x)
mfreqzOS(zOS, pOS, kOS, Fs)

# Plotando o filtro LowPass
figOS, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, x)
ax1.set_title('Sinal Original')
ax2.plot(t, yOS)
ax2.set_title('Sinal Filtrado - One-Seg')
plt.tight_layout()
plt.show()

newyOS = np.array_split(yOS, 100)
for i in range(len(newyOS)):
    newyOS[i] = np.append((newyOS[i]),i)
    newyOS[i] = np.insert((newyOS[i]), 0, 0xAA)

jsonOS = {
    "Name": "One-Seg",
    "ID": (newyOS[0])[0],
    "Frames": len(newyOS),
    "Start": (newyOS[0])[len(newyOS[2]) - 1 ],
    "End": (newyOS[len(newyOS) - 1])[len(newyOS[2]) - 1 ],
}

file = open(saidaOS, 'w')
json.dump(jsonOS, file)
file.close()

# Filtro BANDPASS
ySD = sosfilt(sosSD,x)
mfreqzSD(zSD, pSD, kSD, Fs)
wzSD, hzSD = signal.freqz(zSD, pSD, kSD)

# Plotando o filtro BandPass
figSD, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, x)
ax1.set_title('Sinal Original')
ax2.plot(t, ySD)
ax2.set_title('Sinal Filtrado - SD')
plt.tight_layout()
plt.show()

newySD = np.array_split(ySD, 100)
for i in range(len(newySD)):
    newySD[i] = np.append((newySD[i]),(i + 100))
    newySD[i] = np.insert((newySD[i]), 0, 0xAF)

jsonSD = {
    "Name": "Standart Definition",
    "ID": (newySD[0])[0],
    "Frames": len(newySD),
    "Start": (newySD[0])[len(newySD[2]) - 1 ],
    "End": (newySD[len(newySD) - 1])[len(newySD[2]) - 1 ],
}

file = open(saidaSD, 'w')
json.dump(jsonSD, file)
file.close()

# Filtro HIGHPASS
yHD = sosfilt(sosHD,x)
mfreqzHD(zHD, pHD, kHD, Fs)

# Plotando o filtro HighPass
figHD, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, x)
ax1.set_title('Sinal Original')
ax2.plot(t, yHD)
ax2.set_title('Sinal Filtrado - HD')
plt.tight_layout()
plt.show()

newyHD = np.array_split(yHD, 100)
for i in range(len(newyHD)):
    newyHD[i] = np.append((newyHD[i]),(i + 200))
    newyHD[i] = np.insert((newyHD[i]), 0, 0xFA)

jsonHD = {
    "Name": "High Definition",
    "ID": (newyHD[0])[0],
    "Frames": len(newyHD),
    "Start": (newyHD[0])[len(newyHD[2]) - 1 ],
    "End": (newyHD[len(newyHD) - 1])[len(newyHD[2]) - 1 ],
}

file = open(saidaHD, 'w')
json.dump(jsonHD, file)
file.close()
