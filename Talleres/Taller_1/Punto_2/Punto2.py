import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

#Ruta relativa del archivo
ruta_archivo = os.path.join(os.path.dirname(__file__), 'Data2', 'hysteresis.dat')
datos = pd.read_csv(ruta_archivo, delimiter="\t")

