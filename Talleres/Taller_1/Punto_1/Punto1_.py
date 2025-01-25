import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit
#cargamos y guardamos los datos de intensidad y longitud de onda 
data = pd.read_csv(r"C:\Users\sebas\OneDrive\Documentos\Metodos_computacionales2\metodos2_grupo1\Talleres\Taller_1\Punto_1\Data\Rhodium.csv")
lambda_=data["Wavelength (pm)"].tolist()
intensidad= data["Intensity (mJy)"].tolist()


#Creamos la función para limpiar los datos 
def filtrar(x,y,tolerancia=0.0060):
    x_nuevo= []
    y_nuevo= []
    n= len(x)
    for i in range (0,n-1):
        dato_estudio=y[i]
        dato_posterior=y[i+1]
        resta= abs(dato_estudio-dato_posterior)
        if resta < tolerancia:
            x_nuevo.append(x[i])
            y_nuevo.append(y[i])
    datos_elimininados = len(x)-len(x_nuevo)
    return x_nuevo,y_nuevo, print(f"1.a): Datos eliminados:{datos_elimininados}")

lambda_filtro,intensidad_filtro,D_eliminado= filtrar(lambda_,intensidad)
lambda_filtro=np.array(lambda_filtro)
intensidad_filtro = np.array(intensidad_filtro)

#Graficamos la comparación entre los datos limpiados y los datos originales 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax2.plot(lambda_filtro,intensidad_filtro,label="Datos limpios")
ax2.set_title("Datos limpiados") 
ax2.set_xlabel("Longitud de onda(pm)")           
ax2.set_ylabel("Intensidad(mJy)")                              
ax2.grid(linestyle= "--")   

ax1.plot(lambda_,intensidad,label="Datos originales", linestyle= "-")
ax1.set_title("Datos originales")  
ax1.set_xlabel("Longitud de onda(pm)")           
ax1.set_ylabel("Intensidad(mJy)")                                
ax1.grid(linestyle= "--") 

plt.tight_layout()
plt.show()

#Punto 1.b

#Definimos nuestra función de ajuste 
def gaussianas(x, A1, mu1, sigma1,A2,mu2,sigma2):
    gauss1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    gauss1 = gauss1/ ((2*np.pi)**(1/2)*sigma1)
    gauss2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    gauss2 = gauss2 / ((2*np.pi)**(1/2)*sigma2)
    return gauss1+ gauss2

#Hacemos la separacion del fondo de los picos 
corte_picos = ((lambda_filtro< 75) |  
    ((lambda_filtro > 90) & (lambda_filtro < 95)) |  
    (lambda_filtro > 105) )

#Aplicamos el filtro 
lambda_fondo= lambda_filtro[corte_picos]
intensidad_fondo= intensidad_filtro[corte_picos]

#Hacemos el ajuste 
p0 = [0.1, 20, 20,0.04,200,55]
parametros, _ = curve_fit(gaussianas, lambda_fondo, intensidad_fondo, p0=p0)
ajuste_fondo= gaussianas(lambda_filtro, *parametros)
intensidad_sin_fondo= intensidad_filtro-ajuste_fondo

fig = plt.subplots(2, 1, figsize=(8, 8))
plt.plot(lambda_filtro,intensidad_filtro,label="Datos filtrados")
plt.plot(lambda_filtro,ajuste_fondo,label="Ajuste datos de fondo")
plt.plot(lambda_filtro,intensidad_sin_fondo,label="Aislamiento de picos")
plt.xlabel("Longitud de onda(pm)")
plt.ylabel("Intensidad(mJy)")
plt.title("Ajuste del fondo con Gaussianas")
plt.grid(linestyle="--")
plt.legend()
plt.show()


print(f"1.b) Método: Ajuste gaussiano al fondo y resta con los datos filtrados ")

#Punto 1.c
