import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks

#Ruta relativa del archivo
ruta_archivo = os.path.join(os.path.dirname(__file__), 'Data2', 'hysteresis.dat')
datos = pd.read_csv(ruta_archivo, delimiter="\t")

"Punto 2a"

datos_c = datos.copy()
datos_c.columns = ['tiempo-B-H']
datos_d = datos_c['tiempo-B-H'].str.split(r'[ -]', expand=True)
datos_d.columns = ['tiempo', 'B', 'H']
datos_d['tiempo'] = pd.to_numeric(datos_d['tiempo'], errors='coerce')
datos_d['B'] = pd.to_numeric(datos_d['B'], errors='coerce')
datos_d['H'] = pd.to_numeric(datos_d['H'], errors='coerce')

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.scatter(datos_d['tiempo'], datos_d['B'])
plt.title('Campo magnético vs tiempo')
plt.xlabel('tiempo (ms)')
plt.ylabel('Campo magnético (mT)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(datos_d['tiempo'], datos_d['H'])
plt.grid(True)
plt.title('Densidad del campo interno vs tiempo')
plt.xlabel('tiempo (ms)')
plt.ylabel('Densidad del campo interno (A/m)')

# Ajustar el espaciado entre las subtramas
plt.subplots_adjust(hspace=0.5) # Puedes ajustar el valor 0.5 según tus necesidades

# Ruta relativa para guardar el archivo PDF en la carpeta punto_2
ruta_guardar_pdf = os.path.join(os.path.dirname(__file__), 'histerico.pdf')

# Guardar el archivo PDF en la carpeta punto_2
plt.savefig(ruta_guardar_pdf)
plt.show()

"Punto 2b"

datos_e = datos_d[['tiempo', 'B']].dropna()
indicesB, _ = find_peaks(datos_e['B'].to_numpy(), height=2.43)

w = []
for a in indicesB:
    w.append(datos_e['tiempo'].to_numpy()[a])
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

# Guardar como PDF
plt.savefig(ruta_guardar_pdf, format="pdf", bbox_inches='tight')
plt.close()

"Punto 2c"

