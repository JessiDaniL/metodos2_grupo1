#Librerias
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import simpson

"Punto 1"

#Definimos los linspace para que el circulo quede mas grande
N=300
x= np.linspace(-1.1,1.1,N)
y= np.linspace(-1.1,1.1,N)

#Creamos la grilla de puntos dentro del circulo
X,Y= np.meshgrid(x,y)
mascara= X**2 + Y**2 >= 1

#Definimos nuestra condición adentro 
phi= np.random.rand(N,N)

theta= np.arctan2(Y,X)
phi_borde= np.sin(7*theta)


#Aplicamos la condición a los puntos sobre el borde del circulo
phi[mascara]= phi_borde[mascara]

#Ahora vamos a implementar el metodo para solucionar la ecuacion de poisson
@njit
def poisson(phif,X,Y,mascara,N,tol=1e-4,iter_max=5000):
    h= abs(X[0,1]-X[0,0])
    for e in range(iter_max):
        phi_n= np.copy(phif)
        for i in range(1,N-1):
            for j in range(1,N-1):
                if mascara[i,j]:
                    phi_n[i,j]=0.25*(phif[i+1,j]+phif[i-1,j]+phif[i,j+1]+phif[i,j-1]+4*np.pi*(X[i,j]+Y[i,j])*h**2)
        error= np.max(np.abs(phi_n-phif))
        #Modificamos el array para la siguiente iteración
        for i in range(N):
            for j in range(N):
                phif[i, j] = phi_n[i, j]
                
        if error < tol:
            break
    return phif

mascara_dentro= X**2 + Y**2 < 1

phi_x=poisson(phi,X,Y,mascara_dentro,N)

fig = plt.figure(figsize=(12, 6))

R= X**2+Y**2

#  Contorno 2D 
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(phi, extent=[-1.1, 1.1, -1.1, 1.1], cmap="magma")
ax1.contour(X, Y, R, levels=[1], colors="black",origin="lower", linestyles="dashed")
fig.colorbar(im, ax=ax1, label="φ(x,y)")
ax1.set_title("Condiciones de frontera")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

#Solución 3D
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, phi_x, cmap='magma', edgecolor='none')
ax2.set_title("Solución en 3D")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Phi")

plt.tight_layout()  

#Guardar archivo
ruta_guardar=  ('Talleres/Taller_3b/1.png')
plt.savefig(ruta_guardar, format="png", dpi=300, bbox_inches="tight")
plt.clf()


"Punto 2"

#Datos
L = 2.0
dx = 0.02
c = 1.0
dt = 0.005
Nt = 400

s = (c*dt/dx)**2

def inicial(x):
  return np.exp(-125 * (x-1/2) **2)

x = np.arange(0,L,dx)

condiciones = ["Dirichlet", "Neumann", "Periódica"]

fig, axes = plt.subplots(3, 1, figsize=(6, 9))
lines = []

datos = []
colores = ['#800080', '#FF69B4', '#ADD8E6']
for ax, condicion, color in zip(axes, condiciones,colores):
    u_pre = inicial(x)
    u_pas = np.copy(u_pre)
    u_fut = np.zeros_like(u_pre)
    datos.append([u_pas, u_pre, u_fut])

    line, = ax.plot(x, u_pre, label=condicion, color = color)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, L)
    ax.set_xlabel("Posición")
    ax.set_ylabel("Amplitud")
    ax.legend()
    lines.append(line)

@njit(cache=True)
def actualizar(u_pas, u_pre, u_fut, tipo):
    for i in range(1, len(u_pre) - 1): #Aplicar la derivada central
        u_fut[i] = 2 * u_pre[i] - u_pas[i] + s * (u_pre[i+1] - 2*u_pre[i] + u_pre[i-1])

    # Aplicar condiciones de frontera
    if tipo == "Dirichlet":
        u_fut[0] = 0
        u_fut[-1] = 0
    elif tipo == "Neumann":
        u_fut[0] = u_pre[1]
        u_fut[-1] = u_pre[-2]
    elif tipo == "Periódica":
        u_fut[0] = u_pre[-2]
        u_fut[-1] = u_pre[1]

    return u_pre.copy(), u_fut.copy()

# Crear la animación
def update(frame):
    global datos
    datos_actualizados = []
    for j, (tipo, (u_pas, u_pre, u_fut)) in enumerate(zip(condiciones, datos)):
        u_pas_nuevo, u_pre_nuevo = actualizar(u_pas, u_pre, u_fut, tipo)
        datos_actualizados.append([u_pas_nuevo, u_pre_nuevo, np.copy(u_fut)])
        lines[j].set_ydata(u_pre_nuevo)
    datos = datos_actualizados
    return lines

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=dt * 500, blit=True)

# Guardar el video
writer = animation.FFMpegWriter(fps=60)
ani.save(r"Talleres\Taller_3b\2.mp4", writer=writer)

plt.close(fig)

plt.clf()

"Punto 3"

#3.a

t = 2
alpha = 0.022  # Coeficiente dispersivo
dt = 0.0001  # Paso temporal reducido para evitar inestabilidad
n_p = 200  # Número de puntos espaciales
n_t = int(t/dt)  # Tiempo total de la simulación
dp = 2 / n_p  # Resolución espacial

# Malla espacial y matriz de soluciones
x = np.linspace(0, 2, n_p)
matriz = np.zeros((n_t, n_p))

# Condición inicial suave para evitar valores abruptos
matriz[0] = np.cos(np.pi * x)

# Primer paso temporal asegurando estabilidad numérica
def primer_paso(matriz, dt, dp, alpha, n_p):
    nueva_fila = np.zeros(n_p)
    for i in range(n_p):
        i_p1, i_m1 = (i + 1) % n_p, (i - 1) % n_p
        i_p2, i_m2 = (i + 2) % n_p, (i - 2) % n_p

        matriz_i = matriz[0, i]
        matriz_ip1, matriz_im1 = matriz[0, i_p1], matriz[0, i_m1]
        matriz_ip2, matriz_im2 = matriz[0, i_p2], matriz[0, i_m2]

        # Término no lineal con factor de 1/3 corregido
        no_lin = (1/3) * (matriz_ip1 + matriz_i + matriz_im1) * (matriz_ip1 - matriz_im1)
        
        # Término dispersivo
        dispersivo = matriz_ip2 - 2 * matriz_ip1 + 2 * matriz_im1 - matriz_im2
        
        # Actualización con los mismos factores que el código funcional
        nueva_fila[i] = matriz_i - (dt/dp) * no_lin - (alpha**2) * (dt / dp**3) * dispersivo
    
    return nueva_fila

matriz[1] = primer_paso(matriz, dt, dp, alpha, n_p)

@njit  # Aceleración con Numba para mejorar el rendimiento
def iterar(matriz, n_t, n_p, dt, dp, alpha):
    for n in range(1, n_t - 1):
        for i in range(n_p):
            i_p1, i_m1 = (i + 1) % n_p, (i - 1) % n_p
            i_p2, i_m2 = (i + 2) % n_p, (i - 2) % n_p

            matriz_i = matriz[n-1, i]  # Corrección: usar n-1 en vez de n
            matriz_ip1, matriz_im1 = matriz[n, i_p1], matriz[n, i_m1]
            matriz_ip2, matriz_im2 = matriz[n, i_p2], matriz[n, i_m2]

            # Término no lineal con factor corregido
            no_lin = (1/3) * (matriz_ip1 + matriz[n, i] + matriz_im1) * (matriz_ip1 - matriz_im1)

            # Término dispersivo
            dispersivo = matriz_ip2 - 2 * matriz_ip1 + 2 * matriz_im1 - matriz_im2

            # Actualización corregida
            matriz[n + 1, i] = matriz_i - (dt/dp) * no_lin - (alpha ** 2) * (dt / dp**3) * dispersivo

    return matriz


# Ejecutar la iteración
matriz = iterar(matriz, n_t, n_p, dt, dp, alpha)

# Graficar la evolución de la onda
plt.figure(figsize=(10, 3))
tiempos = np.linspace(0, t, n_t)
plt.imshow(matriz.T, aspect='auto', cmap='plasma', origin='lower',
           extent=[tiempos.min(), tiempos.max(), x.min(), x.max()])
plt.colorbar(label=rf'$\Psi (t,x)$')
plt.xlabel('Time (s)')
plt.ylabel('Angle x (m)')
plt.title('Solución')
#plt.show() #La solución sí converge

# Animación

fig = plt.figure()
ax = plt.axes(xlim=(0,2), ylim=(np.min(matriz), np.max(matriz)))
punto, = ax.plot(x,matriz[0], 'b')

def animacion(frame):
    punto.set_ydata(matriz[frame])
    return punto

ani = animation.FuncAnimation(fig, animacion, frames=range(0, len(matriz),200), interval=50, blit=False)
ani.save('3.a.mp4', writer='ffmpeg', fps=30)

# 3.b

# Calcular cantidades conservadas con scipy.integrate.simps
tiempos = np.linspace(0, t, n_t)
masa = simpson(matriz, x, axis=1)
momento = simpson(matriz * np.gradient(matriz, dp, axis=1), x, axis=1)
energia = simpson((1/3) * matriz**3 - (alpha * np.gradient(matriz, dp, axis=1))**2, x, axis=1)

# Graficar cantidades conservadas
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].plot(tiempos, masa, label='Masa')
axs[0].set_ylabel('Masa')
axs[0].grid()

axs[1].plot(tiempos, momento, label='Momento', color='r')
axs[1].set_ylabel('Momento')
axs[1].grid()

axs[2].plot(tiempos, energia, label='Energía', color='g')
axs[2].set_ylabel('Energía')
axs[2].set_xlabel('Tiempo')
axs[2].grid()

plt.tight_layout()
ruta_guardar_pdf =  ('3_b.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')
plt.close(fig)

"Punto 4"

c4=0.5
#tiempo de la animacion:
T4=10
x4=np.linspace(0,2,int(2/0.01))
y4 = np.linspace(0,1,int(1/0.01))
#definiendo las condiciones de frontera:
def condiciones1(x):
#para los límites:
    x[0,:] = 0
    x[-1,:] = 0
    x[:,0] = 0
    x[:,-1] = 0
#para los límites en la pared:
    limitex1=int((1-0.02)/0.01)
    limitex2=int((1+0.02)/0.01)
    limitey1=int(0.42/0.01)
    limitey2=int((1-0.42)/0.01)
    x[limitex1-1:limitex2+1,:limitey1+1]=0
    x[limitex1-1:limitex2+1,limitey2-1:]=0
    return x

onda=np.zeros((int(10/0.005), int(2/0.01), int(1/0.01)))

for i in range(int(2/0.01)):
    for j in range(int(1/0.01)):
        onda[0,i,j]=2*np.exp(-150*(((x4[i]-2/5)**2)+((y4[j]-1/2)**2)))
onda[0]=condiciones1(onda[0])
for i in range(1, int(2/0.01)-1):
    for j in range(1, int(1/0.01)-1):
        onda[1,i,j] = onda[0,i,j]+(1/2)*(c4*0.005)**2*((onda[0, i+1, j]-2*onda[0,i,j]+onda[0,i-1,j])/0.01**2 + (onda[0, i, j+1]-2*onda[0, i, j] + onda[0, i, j-1])/0.01**2)
onda[1]=condiciones1(onda[1])

# Definir la lente
lalente=np.zeros((int(2/0.01), int(1/0.01)), dtype=bool)
lente_center_xce=int(1/0.01)
lente_center_y=int(0.5/0.01)
lente_radius_x=int(0.2/0.01)
lente_radius_y=int(0.1/0.01)
#para definir qué regiones del espacio estan en la lente:
for k in range(int(2/0.01)):
    for l in range(int(1/0.01)):
        if ((((x4[k]-1)**2)/(lente_radius_x**2)) + (y4[l]-(1/2))**2/(lente_radius_y**2))<= 1:
            lalente[k,l]=True

c_lente=c4/5 # Velocidad de la onda en la lente
limitex1=int((1-0.02)/0.01)
limitex2=int((1+0.02)/0.01)
limitey1=int(0.42/0.01)
limitey2=int((1-0.42)/0.01)

def evolucion(onda, iter):
    for n in range(1, iter): #numero de iteraciones en el tiempo
        for i in range(1, int(2/0.01)-1): #iteraciones en x
            for j in range(1, int(1/0.01)-1): #iteraciones en y
                if lalente[i, j]:
                    c=c_lente
                else:
                    c=c4
                onda[n+1, i, j] = 2*onda[n, i, j]- onda[n-1, i, j] + (c*0.005)**2 * ((onda[n, i+1, j] - 2*onda[n, i, j] + onda[n, i-1, j]) / 0.01**2 + (onda[n, i, j+1] - 2*onda[n, i, j] + onda[n, i, j-1]) / 0.01**2)
        onda[n+1]=condiciones1(onda[n+1])
    return onda
onda=evolucion(onda, int(10/0.005)-1)
fig, ax=plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
mask=np.zeros((int(2/0.01),int(1/0.01)),dtype=bool)
mask[limitex1:limitex2,:limitey1]=True
mask[limitex1:limitex2,limitey2:]=True
cmap = ax.imshow(onda[0], extent=[0, 2, 0, 1], origin='lower', cmap='inferno', vmin=np.min(onda), vmax=np.max(onda))
fig.colorbar(cmap)

# Dibujar el lente
lente_image=np.ma.masked_where(~lalente, np.ones((int(2/0.01),int(1/0.01))))
ax.imshow(lente_image, extent=[0, 2, 0, 1], origin='lower', cmap='Blues', alpha=0.3)

def draw_frame(frame):
    frame_data = onda[frame].copy()
    frame_data[mask] = np.nan
    cmap.set_array(frame_data)
    return [cmap]
anim4 = animation.FuncAnimation(fig, draw_frame, frames=range(0, len(onda), 10), interval=50, blit=False)
anim4.save("4_a.mp4", writer="ffmpeg", fps=20)