import numpy as np
import matplotlib.pyplot as plt
import time

# Inputs (Modify according to requirements)

TotalGasGravity = 0.65 + 9 / 50  # Total gas gravity

# Mole Fractions
N2 = 2 / 100
CO2 = 8 / 100
H2S = 8 / 100

# Pressure (psi)
Pwh = 190  # Pressure at wellhead
Pbh = 2600  # Pressure at bottomhole

# Temperature (R)
Twh_F = 100  # Temperature at wellhead (Fahrenheit)
Twh = Twh_F + 460  # Temperature at wellhead (Rankine)
Tbh_F = Twh + 70  # Temperature at bottomhole (Fahrenheit)
Tbh = Tbh_F + 460  # Temperature at bottomhole (Rankine)

# Depth (ft)
Depth = 2800  # Depth

# Gradients
Grad_T = (Tbh - Twh) / 99
Grad_D = Depth / 99
Grad_P = (Pbh - Pwh) / 99

# Universal Gas Constant
R = 10.73

# Standing Pseudocritical HC Gravity, Temperature & Pressure calculation
sg_hc = (TotalGasGravity - (34 * H2S / 29) - (28 * N2 / 29) - (44 * CO2 / 29)) / (1 - N2 - CO2 - H2S)
Tpc = 168 + 325 * sg_hc - 12.5 * sg_hc * sg_hc
Ppc = 677 + 15 * sg_hc - 37.5 * sg_hc * sg_hc

# Pseudocritical Temperature & Pressure of the Gas Mixture from Kay's Mixing Rule
Tpcc = (1 - H2S - CO2 - N2) * Tpc + 227.5 * N2 + 547.9 * CO2 + 672.4 * H2S
Ppcc = (1 - H2S - CO2 - N2) * Ppc + 493.1 * N2 + 1071 * CO2 + 1306 * H2S

# Corrected Pseudocritical Temperature & Pressure with Wichert & Aziz Method
A = CO2 + H2S
B = H2S
E = 120 * (A**0.9 - A**1.6) + 15 * (B**0.5 - B**4)
Tpc_correct = Tpcc - E
Ppc_correct = Ppcc * Tpc_correct / (Tpcc + H2S * (1 - H2S) * E)

# Determining the interval for Temperature, Pressure & Depth
T = np.linspace(Twh, Pbh, 100)
P = np.linspace(Pwh, Pbh, 100)
D = np.linspace(0, Depth, 100)

# Creating Lists
for i in range(100):
    P[i] = (i - 1) * Grad_P + Pwh
    T[i] = (i - 1) * Grad_T + Twh
    D[i] = (i - 1) * Grad_D

# Pseudoreduced Temperature & Pressure
Tpr = T / Tpc_correct
Ppr = P / Ppc_correct

Zfactor = np.zeros(100)
Cg = np.zeros(100)
Ug = np.zeros(100)
Bg = np.zeros(100)
Density = np.zeros(100)
Pseudo_Pressure = np.zeros(100)

for j in range(100):
    # Determination of Gas Compressibility Factor with Redlich-Kwong
    A1 = (0.4278 * Ppr[j]) / (Tpr[j]**(2.5))
    B1 = (0.0867 * Ppr[j]) / (Tpr[j])
    C_ = [1, -1, (A1 - B1 - B1**2), -A1 * B1]
    z = np.roots(C_)
    Zfactor[j] = np.real(z[0])

    if j != 0:
        Cg[j] = (1 / P[j - 1]) - (1 / Zfactor[j - 1]) * ((Zfactor[j] - Zfactor[j - 1]) / (P[2] - P[1]))

for n in range(100):
    # Molecular Weight
    MW = 29 * sg_hc

    # Density calculation (lb/ft3)
    Density[n] = (P[n] * MW) / ((T[n] + 460) * R * Zfactor[n])

    # Viscosity calculation with Lee-Gonzales and Eaken Correlation
    X = 3.5 + 986 / T[n] + 0.01 * MW
    Y = 2.4 - 0.2 * X
    K = (9.4 + 0.02 * MW) * T[n]**1.5 / (209 + 19 * MW + T[n]) * 10**-4
    Ug[n] = K * (np.exp(X * (Density[n] / 62.4)**Y))

    # Gas Formation Volume Factor (rcf/scf)
    Bg[n] = 0.02827 * Zfactor[n] * T[n] / P[n]

    # Real Gas Pseudo Pressure Calculation by Trapezoid Rule
    total = 0
    for y in range(n):
        total += 2 * P[y] / (Ug[y] * Zfactor[y])

    t = (P[n] - P[0]) / (2 * n)
    Pseudo_Pressure[n] = t * total

# Results and Plots

# Z-Factor vs Pressure Graph
plt.figure(1)
plt.plot(P, Zfactor, 'r')
plt.ylabel('Z-Factor')
plt.xlabel('Pressure (psi)')
plt.title('Z-Factor vs Pressure')

# Z-Factor vs Temperature Graph
plt.figure(2)
plt.plot(T, Zfactor, 'r')
plt.ylabel('Z-factor')
plt.xlabel('Temperature (R)')
plt.title('Z-Factor vs Temperature')

# Z-Factor vs Depth Graph
plt.figure(3)
plt.plot(D, Zfactor, 'r')
plt.ylabel('Z-factor')
plt.xlabel('Depth (ft)')
plt.title('Z-Factor vs Depth')

# Formation Volume Factor vs Pressure Graph
plt.figure(4)
plt.plot(P, Bg, 'g')
plt.ylabel('Formation Volume Factor (rcf/scf)')
plt.xlabel('Pressure (psi)')
plt.title('Gas Formation Volume Factor vs Pressure')

# Formation Volume Factor vs Temperature Graph
plt.figure(5)
plt.plot(T, Bg, 'g')
plt.ylabel('Formation Volume Factor (rcf/scf)')
plt.xlabel('Temperature (R)')
plt.title('Gas Formation Volume Factor vs Temperature')

# Formation Volume Factor vs Depth Graph
plt.figure(6)
plt.plot(D, Bg, 'g')
plt.ylabel('Formation Volume Factor (rcf/scf)')
plt.xlabel('Depth (ft)')
plt.title('Gas Formation Volume Factor vs Depth')

# Compressibility Factor vs Pressure
plt.figure(7)
plt.plot(P, Cg)
plt.ylabel('Compressibility (1/psi)')
plt.xlabel('Pressure (psi)')
plt.title('Compressibility vs Pressure')

# Compressibility Factor vs Temperature
plt.figure(8)
plt.plot(T, Cg)
plt.ylabel('Compressibility (1/psi)')
plt.xlabel('Temperature (R)')
plt.title('Compressibility vs Temperature')

# Compressibility Factor vs Depth
plt.figure(9)
plt.plot(D, Cg)
plt.ylabel('Compressibility (1/psi)')
plt.xlabel('Depth (ft)')
plt.title('Compressibility vs Depth')

# Viscosity vs Pressure Graph
plt.figure(10)
plt.plot(P, Ug, 'b')
plt.ylabel('Viscosity (cP)')
plt.xlabel('Pressure (psi)')
plt.title('Viscosity vs Pressure')

# Viscosity vs Temperature Graph
plt.figure(11)
plt.plot(T, Ug, 'b')
plt.ylabel('Viscosity (cP)')
plt.xlabel('Temperature (R)')
plt.title('Viscosity vs Temperature')

# Viscosity vs Depth Graph
plt.figure(12)
plt.plot(D, Ug, 'b')
plt.ylabel('Viscosity (cP)')
plt.xlabel('Depth (ft)')
plt.title('Viscosity vs Depth')

# Density vs Pressure Graph
plt.figure(13)
plt.plot(P, Density, 'm')
plt.ylabel('Density (lbm/ft3)')
plt.xlabel('Pressure (psi)')
plt.title('Density vs Pressure')

# Density vs Temperature Graph
plt.figure(14)
plt.plot(T, Density, 'm')
plt.ylabel('Density(lbm/ft3)')
plt.xlabel('Temperature (R)')
plt.title('Density vs Temperature')

# Density vs Depth Graph
plt.figure(15)
plt.plot(D, Density, 'm')
plt.ylabel('Density(lbm/ft3)')
plt.xlabel('Depth (ft)')
plt.title('Density vs Depth')

# Pseudo Pressure vs Pressure Graph
plt.figure(16)
plt.plot(P, Pseudo_Pressure, 'k')
plt.ylabel('Pseudo Pressure (psi^2/c)')
plt.xlabel('Pressure (psi)')
plt.title('Pseudo Pressure vs Pressure')

# Pseudo Pressure vs Temperature Graph
plt.figure(17)
plt.plot(T, Pseudo_Pressure, 'k')
plt.ylabel('Pseudo Pressure (psi^2/cP)')
plt.xlabel('Temperature (R)')
plt.title('Pseudo Pressure vs Temperature')

# Pseudo Pressure vs Depth Graph
plt.figure(18)
plt.plot(D, Pseudo_Pressure, 'k')
plt.ylabel('Pseudo Pressure (psi^2/cP)')
plt.xlabel('Depth (ft)')
plt.title('Pseudo Pressure vs Depth')

plt.show()

time.sleep(1)
