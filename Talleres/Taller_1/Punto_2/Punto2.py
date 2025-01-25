import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
import os

#Carga de datos en formato de lista

with open('Talleres\Taller_1\Punto_2\Data2/hysteresis.dat', 'r') as file:
    lines = file.readlines() 

# Organización de los datos en listas
tiempo = []
B = []
H = []

for line in lines:
    corregido = ""
    i = 0
    while i < len(line):

        # Separar números negativos
        if line[i] == '-' and i > 0 and line[i-1] not in [' ', '-']:
            corregido += " -"

        # Separar números "0."
        elif line[i:i+2] == '0.' and i > 0 and line[i-1] not in [' ', '-']:
            corregido += " 0."
            i += 1
        else:
            corregido += line[i]
        i += 1

    numeros = list(filter(None, corregido.split()))
    
    # Convertir a float y asignar a listas
    try:
        tiempo.append(float(numeros[0]))
        B.append(float(numeros[1]))
        H.append(float(numeros[2]))
    except (ValueError, IndexError) as e:
        print(f"Error al procesar la línea: {line.strip()} -> {e}")

"Punto 2a"

fig_size = (10, 8)
plt.figure(figsize=fig_size)
plt.subplot(2, 1, 1)
plt.scatter(tiempo, B)
plt.title('Campo magnético vs tiempo')
plt.xlabel('tiempo (ms)')
plt.ylabel('Campo magnético (mT)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(tiempo, H)
plt.grid(True)
plt.title('Densidad del campo interno vs tiempo')
plt.xlabel('tiempo (ms)')
plt.ylabel('Densidad del campo interno (A/m)')
plt.subplots_adjust(hspace=0.5)

# Guardar el archivo PDF en la carpeta punto_2
ruta_guardar_pdf = os.path.join(os.path.dirname(__file__), 'histerico.pdf')
plt.savefig(ruta_guardar_pdf)

"Punto 2b"

indicesB, _ = find_peaks(B, height=2.43)

w = []
for a in indicesB:
    w.append(tiempo[a])
del w[-2:]

periodo = (w[-1] - w[0]) * 1000 / (len(w) - 1)
frecuencia = 1 / periodo

texto = f"""
Frecuencia: {frecuencia} Hz

Se usó una función llamada find_peaks para obtener las posiciones de los picos de la gráfica.
Se hizo una gráfica y se observó que los últimos valores no tomaban un pico, así que se recortó la lista.
Dado que entre dos picos hay una onda completa, se tomó el tiempo entre el primer y último pico y
se dividió entre el número de picos (los valores de la lista) menos 1, y se sacó la inversa.
"""

plt.figure(figsize=(8.5, 11))
plt.axis('off')
plt.text(0.5, 0.8, texto, fontsize=12, ha='center', va='top', wrap=True, linespacing=1.5)

# Ruta relativa para guardar el archivo PDF en la carpeta punto_2
ruta_guardar_pdf = os.path.join(os.path.dirname(__file__), 'frecuencia_resultados.pdf')
plt.savefig(ruta_guardar_pdf, format="pdf", bbox_inches='tight')
plt.close()

"Punto 2c"

#Graficar H vs B

fig_size = (10, 8)
plt.figure(figsize=fig_size)
plt.scatter(H, B)
plt.title('Densidad del campo interno vs campo magnético')
plt.ylabel('Densidad del campo interno (A/m)')
plt.xlabel('Campo magnético (mT)')
plt.grid(True)

plt.show()
