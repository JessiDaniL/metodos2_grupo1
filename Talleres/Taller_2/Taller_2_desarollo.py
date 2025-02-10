import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mode
import pandas as pd
from typing import List
from numpy.typing import NDArray
import matplotlib.dates as mdates

"""

El tiempo de procesamiento del código se midió con cronómetro. El tiempo estimado de espera es de 4 minutos para un computador
AMD Ryzen 5 3600 6 - Core (equivalente en Intel es Intel Core i5-11600K) con una memoria RAM de 16 GB.

"""

"Punto 1.a"

#Cargamos la funcion generadora de datos
from numpy.typing import NDArray
def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
 frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
 ts = np.arange(0.,t_max,dt)
 ys = np.zeros_like(ts,dtype=float)
 for A,f in zip(amplitudes,frecuencias):
  ys += A*np.sin(2*np.pi*f*ts)
  ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
 return ts,ys

#1.a) Implementamos la transformada explicita de fourier 
def Fourier(t, y, f:float) -> complex:
  F=0
  for i in range(len(t)):
    F += y[i]*np.exp(-2j*np.pi*f*t[i])
  return F

#Generamos los datos para dos señales, una con ruido y otra sin 
señal_sin_ruido = datos_prueba(15,0.1,[1,1.5,2],[1,.614,2.8])
señal_con_ruido = datos_prueba(15,0.1,[1,1.5,2],[1,.614,2.8],ruido=0.8)
ts= señal_sin_ruido[0]
ys= señal_sin_ruido[1]

ts_ruido = señal_con_ruido[0]
ys_ruido = señal_con_ruido[1]

freq= np.linspace(0,5,500)
FFT= [Fourier(ts,ys,f) for f in freq]
FFT_ruido = [Fourier(ts_ruido,ys_ruido,f) for f in freq]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

ax1.plot(freq,np.abs(FFT),c="b",label="FFT señal sin ruido")
ax1.set_xlabel("Frecuencia")
ax1.set_ylabel("FFT")
ax1.legend()

ax2.plot(freq,np.abs(FFT_ruido),c="r",label="FFT señal con ruido")
ax2.set_xlabel("Frecuencia")
ax2.set_ylabel("FFT")
ax2.legend()

plt.tight_layout()

ruta_guardar_pdf =  ('Talleres/Taller_2/1.a.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')



print ("1.a) Cuando el valor del ruido es igual o mayor al valor de la amplitud de una las frecuencias no se puede distingir la frecuencia asociada a esa amplitud  ")

"1.b"

#Creamos la señal
señal_fundamental = datos_prueba(10,0.01,[2],[4])
ts_fund = señal_fundamental[0]
ys_fund= señal_fundamental[1]

freq_fund= np.linspace(0,5,500)
FFT_fund= [Fourier(ts_fund,ys_fund,f) for f in freq_fund]
FTT_fund_abs= np.abs(FFT_fund)

def gaussiana(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def ajuste_racional(x, a, b, c):
    return a / (x + b) + c

def FWHM (f,FFT_abs):
    mascara = (f> 4- 0.06) & ( f< 4+ 0.06)
    freq_fit = f[mascara]
    FFT_fit = FFT_abs[mascara]
    p, _ = curve_fit(gaussiana, freq_fit, FFT_fit, p0=[max(FFT_abs),4, 0.02])
    A, mu, sigma = p
    FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma
    return FWHM

#Ahora tenemos que ver la tendencia cambiando t_max
t_max = np.linspace(10,300,15)
FWHM_x= []
for i in t_max:
    señal_x = datos_prueba(i,0.01,[2],[4])
    ts_x = señal_x[0]
    ys_x= señal_x[1]
    freq_x= np.linspace(0,5,500)
    FFT_fundx= [Fourier(ts_x,ys_x,f) for f in freq_x]
    FTT_absx= np.abs(FFT_fundx)
    valor=FWHM(freq_x,FTT_absx)
    FWHM_x.append(valor)

#Agamos el ajuste a los datos 
p_o, _ = curve_fit(ajuste_racional,t_max, FWHM_x, maxfev=5000)
a,b,c = p_o

x_ajuste = np.linspace(np.min(t_max),np.max(t_max),100)
ajuste= ajuste_racional(x_ajuste,a,b,c)

plt.loglog(t_max, FWHM_x, 'o-', label='FWHM vs t_max',color= "b")
plt.loglog(x_ajuste,ajuste,label="Modelo de función racional",color="r")
plt.xlabel("t_max (s)")
plt.ylabel("FWHM")
plt.title("FWHM vs t_max en escala log-log")
plt.legend()
ruta_guardar_pdf =  ('Talleres/Taller_2/1.b.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')



"Punto 1.c"

datos = np.loadtxt("Talleres\Taller_2\Datos/Datos_punto1.csv", delimiter=";")

t = np.array(datos[:, 0])
y = np.array(datos[:, 1])
sigma_y= np.array(datos[:, 2] ) 


#Calculamos la frecuencia de Nyquist
delta_t=np.diff(t)
val_d_t=mode(delta_t)
f_nyquist = 1 / (2 * val_d_t)
print(f"1.c) frecuencia Nyquist: {f_nyquist:.6f}")

#Eliminamos el valor del promedio de la intensidad de los datos
y_filt=y-y.mean()

#Calculamos la transformada de fourier
f=np.arange(0,5,0.005)
FFT_data= np.abs(Fourier(t,y_filt,f))

#hayamos la frecuencia mas alta, que corresponde a la frecuencia dominante de la señal
i_max=np.argmax(FFT_data)
frec_true=round(f[i_max],2)
print(f"1.c) f true: {frec_true}/Día")

phi = np.mod(frec_true* t, 1)

# Graficar y vs. φ, para comprobar el resultado
plt.figure()
plt.scatter(phi, y, s=5, alpha=0.7, label="Datos",c="b")
plt.xlabel("Fase φ")
plt.ylabel("Intensidad y")
plt.title("Intensidad vs. Fase")
ruta_guardar_pdf =  ('Talleres/Taller_2/1.c.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')



"Punto 2"

Hfield=pd.read_csv(r"Talleres\Taller_2\Data\H_field.csv", sep=",")
t=Hfield["t"].tolist()
H=Hfield["H"].tolist()
# Punto 2.a- Comparativa
freq=np.fft.rfftfreq(len(t),t[1]-t[0])
FTT=np.fft.rfft(H)
f_fast=freq[np.argmax(np.abs(FTT))]  #np.argmax(np.abs(FFT)) para obtener la magnitud máxima de FFT

#Ahora para el cálculo de la frecuencia general:
def Fourier(t:NDArray[float], y:NDArray[float], f:float) -> complex:
  F=0
  for i in range(len(t)):
    F+= y[i]*np.exp(-2j*np.pi*f*t[i])
  return F
frecuencias=np.linspace(0,3,500)
FTT2=[]
for a in range(len(frecuencias)):
    FTT2.append(Fourier(t,H,frecuencias[a])) #Crea una lista con la transformada para cada frecuencia
f_general=frecuencias[np.argmax(np.abs(FTT2))]
print(f"2.a) {f_fast = :.5f}; {f_general = }")

#Para hacer las gráficas:
t=np.array(t)
fase_fast=np.mod(f_fast*t,1) 
fase_general=np.mod(f_general*t,1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(fase_fast, H, alpha=0.5, s=5)
plt.xlabel('fase_fast')
plt.ylabel('H')
plt.title('H vs fase_fast')

plt.subplot(1, 2, 2)
plt.scatter(fase_general, H, alpha=0.5, s=5)
plt.xlabel('fase_general')
plt.ylabel('H')
plt.title('H vs fase_general')

plt.tight_layout()
ruta_guardar_pdf =  ('Talleres/Taller_2/2.a.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')


"Punto 2.b -manchas solares"

aavso=pd.read_csv(r"Talleres\Taller_2\Data\list_aavso-arssn_daily.txt",sep=r"\s+",header=None,names=["Year", "Month", "Day", "SSN"])
#con el sep se separa con espacios, y con los demás parámetros se delimitan las columnas
tiempo_aavso=[]
tiempo_normal=[]
SSN=[]
for i in range(len(aavso)):
#El primer if se usó para eliminar las primeras 2 filas que no son datos
    if i>2 and int(aavso.iloc[i,0])<=2012:
        if int(aavso.iloc[i,0])<2012:
          fecha=pd.to_datetime(f"{aavso.iloc[i, 0]}-{aavso.iloc[i, 1]}-{aavso.iloc[i, 2]}",format="%Y-%m-%d")
          tiempo_aavso.append(fecha)
          SSN.append(int(aavso.iloc[i,3]))
          fecha_normal=int(aavso.iloc[i,0])+(int(aavso.iloc[i,1])/12)+(int(aavso.iloc[i,2])/365)
          tiempo_normal.append(fecha_normal)
        else:
          if int(aavso.iloc[i,1])+int(aavso.iloc[i,2])==2:
            fecha=pd.to_datetime(f"{aavso.iloc[i, 0]}-{aavso.iloc[i, 1]}-{aavso.iloc[i, 2]}",format="%Y-%m-%d")
            tiempo_aavso.append(fecha)
            SSN.append(int(aavso.iloc[i,3])) 
            fecha_normal=int(aavso.iloc[i,0])+(int(aavso.iloc[i,1])/12)+(int(aavso.iloc[i,2])/365)
            tiempo_normal.append(fecha_normal)

plt.figure(figsize=(12, 5))  # Ajustar tamaño
plt.scatter(tiempo_aavso, SSN, color="blue", marker="o", alpha=0.5, label="SSN")

plt.xlabel("Fecha")
plt.ylabel("SSN")
plt.title("SSN vs Fecha")
plt.xticks(rotation=45)  # Rotar etiquetas de fecha
#plt.show()

"Punto 2.b.a"

tiempo_normal=np.array(tiempo_normal)
SSN= np.array(SSN)
freq_aavso=np.fft.rfftfreq(len(tiempo_normal), np.mean(np.diff(tiempo_normal)))  
# No se usó tiempo_normal[1]-tiempo_normal[2] como intervalo de muestreo sino el promedio de las diferencias
FTT_aavso=np.fft.rfft(SSN)
maximo_fourier=np.argmax(np.abs(FTT_aavso[1:]))+1  #Se suma 1 para eliminar el pico del valor 0
frecuencia_dominante=freq_aavso[maximo_fourier]
P_solar=frecuencia_dominante**-1
print(f"2.b.a) {P_solar = } años")

"Punto 2.b.b - Prediga el número de manchas solares desde 2012 hasta la fecha de entrega de esta tarea (10 Feb 2025)"

def inversa(FFT_aavso:np.ndarray, frecuencias:np.ndarray, anio:int, mes:int, dia:int): 
    tiempo=(anio-2012)*365+(mes-1)*30+dia  
    tiempo=tiempo%(11*365) # Período del ciclo solar
    # Tomar solo las componentes más significativas
    N = len(FTT_aavso)
    indices = np.argsort(np.abs(FTT_aavso))[-10:]  #indices de menor a mayor magnitud de la transformada_aavso
    primeros_10=indices[-10:] #primeros 10 coeficientes de mayor magnitud
    y_aavso=0
    for a in primeros_10:
     y_aavso+=(FTT_aavso[a] +np.conj(FTT_aavso[a]))*np.exp(2j*np.pi*freq_aavso[a]*tiempo)
    return round((y_aavso.real)/len(FTT_aavso))

n_manchas_hoy=inversa(FTT_aavso, freq_aavso, 2025, 2, 10)
print(f'2.b.b) {n_manchas_hoy = }')

fechas=np.arange(np.datetime64('2012-01-01'),np.datetime64('2025-02-10'),np.timedelta64(30,'D'))

manchas_reconstruidas=[]
for f in fechas:
    w=inversa(FTT_aavso, freq_aavso, f.astype(object).year, f.astype(object).month, f.astype(object).day)
    manchas_reconstruidas.append(w)

manchas_reconstruidas=np.array(manchas_reconstruidas)
plt.figure(figsize=(12, 5))
plt.plot(fechas, manchas_reconstruidas, color="red", label="Reconstrucción Fourier")
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Etiquetas cada 2 años
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("Número de Manchas Solares")
plt.title("Evolución de Manchas Solares (2012-2025)")
plt.legend()
plt.grid()

ruta_guardar_pdf =  ('Talleres/Taller_2/2.b.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')

"Punto 3.a"

#Filtro gaussiano

frecuencias = freq_aavso 

trans = np.fft.fft(SSN)

longitud = len(SSN)

frecuencia = np.fft.fftfreq(longitud, 1)

abs_trans_rapida = np.abs(trans) #Corresponde a la magnitud de la transformada

filtro_gaussiano = trans*np.exp(-(np.abs(frecuencia)*1000)**2)

plt.plot((frecuencia[1:longitud//2]),(abs_trans_rapida[1:longitud//2]),label='Original')

plt.plot((frecuencia[1:longitud//2]),((np.abs(filtro_gaussiano))[1:longitud//2]),label='Filtrada')

plt.ylim(10,)
plt.xscale('log')
plt.yscale('log')
plt.legend()

ruta_guardar_pdf = ('Talleres/Taller_2/3.1.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')

"Punto 3.b"

img_gato = plt.imread("Talleres/Taller_2/Data/catto.png")
img_castillo = plt.imread("Talleres/Taller_2/Data/Noisy_Smithsonian_Castle.jpg")

altura_g,ancho_g = img_gato.shape
altura_c,ancho_c = img_castillo.shape

cmap = plt.get_cmap("grey")
cmap.set_bad((1,0,0))
cmap

#Tranformada rápida de Fourier para visualizar las frecuencias que generan ruido en la imagen del gato

transformada_ima_gato = np.fft.fft2(img_gato)
transformada_movida_gato = np.fft.fftshift(transformada_ima_gato)
magnitud_transformada_movida_gato = np.abs(transformada_movida_gato)
magnitud_logaritmo_gato = np.log1p(magnitud_transformada_movida_gato)
#plt.imshow(magnitud_logaritmo_gato,cmap=cmap)

#Tranformada rápida de Fourier para visualizar las frecuencias que generan ruido en la imagen del castillo

transformada_ima_castillo = np.fft.fft2(img_castillo)
transformada_movida_castillo = np.fft.fftshift(transformada_ima_castillo)
magnitud_transformada_movida_castillo = np.abs(transformada_movida_castillo)
magnitud_logaritmo_castillo = np.log1p(magnitud_transformada_movida_castillo)
#plt.imshow(magnitud_logaritmo_castillo,cmap=cmap)


#Función para filtrar el ruido de acuerdo con el tamaño de las imágenes y la intensa magnitud del brillo; esta es propia de cada imagen.

def filtrar(altura,ancho, magnitud,transformada,tolerancia):

    r= 15

    centro_y = altura// 2
    centro_x = ancho // 2

    indices_puntos_brillantes = np.argwhere(magnitud > (np.max(magnitud)*tolerancia))

    indices_sin_centro = []

    for (h,a) in indices_puntos_brillantes:

        if not (((centro_y-r) < h < (centro_y+r)) and ((centro_x-r) < a < (centro_x+r))):

            indices_sin_centro.append((h,a))

    copia_transformada = transformada.copy()

    for (h,a) in indices_sin_centro:

        copia_transformada[h-5:h+5,a-5:a+5] = 0

    magnitud_copia = np.abs(copia_transformada)

    magnitud_log = np.log1p(magnitud_copia)

    return magnitud_log,copia_transformada


magnitud_log_g,copia_transformada_g = filtrar(altura_g, ancho_g,magnitud_logaritmo_gato, transformada_movida_gato,0.55)
magnitud_log_c,copia_transformada_c = filtrar(altura_c,ancho_c,magnitud_logaritmo_castillo,transformada_movida_castillo,0.75)

#plt.imshow(magnitud_log_g,cmap = cmap)
#plt.imshow(magnitud_log_c,cmap = cmap)


reorganizacion_imagen_castillo = np.fft.ifftshift(copia_transformada_c)
imagen_castillo_final = np.fft.ifft2(reorganizacion_imagen_castillo).real
plt.imshow(imagen_castillo_final,cmap = cmap)
ruta_guardar_castillo_png =  ('Talleres/Taller_2/3.b.a.png')
plt.savefig(ruta_guardar_castillo_png,format="png", bbox_inches='tight')



reorganizacion_imagen_gato= np.fft.ifftshift(copia_transformada_g)
imagen_gato_final = np.fft.ifft2(reorganizacion_imagen_gato).real
plt.imshow(imagen_gato_final,cmap = cmap)
ruta_guardar_gato_png =  ('Talleres/Taller_2/3.b.b.png')
plt.savefig(ruta_guardar_gato_png,format="png", bbox_inches='tight')