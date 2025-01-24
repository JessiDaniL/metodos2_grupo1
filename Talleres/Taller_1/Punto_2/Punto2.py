import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

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