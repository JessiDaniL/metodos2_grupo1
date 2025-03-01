#Librerias
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#Parte a--------------------------------------------------------------------------------------------------


#Punto 2

#Fuerza de Coulomb en unidades atómicas.
#Hay que tener en cuenta que para definir la función proporcionada en la guía, se asume ax y ay como -r/|r^{3}| porque la masa se consideracomo uno

def f_prime(t,f):
    x, y, vx, vy = f

    r = np.sqrt(x**2 + y**2)

    ax = -x/np.abs(r**3)
    ay = -y/np.abs(r**3)

    return np.array([vx,vy,ax,ay])

#Método Runge Kutta (4to orden)
def RK4_step(F,y0,t,dt):
    k1 = F(t,y0)
    k2 = F( t+dt/2, y0 + dt*k1/2 )
    k3 = F( t+dt/2, y0 + dt*k2/2  )
    k4 = F( t+dt, y0 + dt*k3  )
    return y0 + dt/6 * (k1+2*k2+2*k3+k4)

def runge_kutta(F,y_0,ts,dt):

    t_v = np.arange(ts[0],ts[1],dt)
    y_v = np.zeros((len(t_v),len(y_0)))
    y_v[0] =y_0

    for i in range(1, len(t_v)):
        t = t_v[i-1]
        y_v[i] = RK4_step(F, y_v[i-1], t, dt)

    return t_v, y_v

x0,y0,vx0,vy0 = 1.0,0.0,0.0,1.0
condiciones_iniciales= np.array([x0,y0,vx0,vy0])

dt = 0.01
t_s=(0,10)

t_v,y_v = runge_kutta(f_prime,condiciones_iniciales,t_s,dt)


#Solución

x_r,y_r,vx_r,vy_r = y_v[:,0],y_v[:,1],y_v[:,2],y_v[:,3]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('x (unidades del radio de Bohr)')
ax.set_ylabel('y (unidades del radio de Bohr)')
ax.set_title('Simulación de la órbita del electrón')
ax.grid()

line, = ax.plot([], [], 'b-', label='Órbita')
electron, = ax.plot([], [], 'bo', label='Electrón')
ax.scatter(0, 0, color='red', marker='o', label='Protón')
ax.legend()

def update(frame):
    line.set_data(x_r[:frame], y_r[:frame])
    electron.set_data([x_r[frame]], [y_r[frame]])
    return line, electron

ani = animation.FuncAnimation(fig, update, frames=range(0,len(x_r),10), interval=10, blit=True)
ani.save("Talleres/Taller_3/2.a.simulation.mp4", writer="ffmpeg", fps=30)

plt.close(fig)


#Cálculo teórico y experimental del periodo

# Encontrar período simulado midiendo el tiempo entre dos pasos por el mismo punto
def periodo_simulacion(tiempo, posiciones):
    tiempos_de_cruce = []
    for i in range(1, len(posiciones)):
        if posiciones[i-1] > 0 and posiciones[i] < 0:
            tiempos_de_cruce.append(tiempo[i])
    if len(tiempos_de_cruce) >= 2:
        return tiempos_de_cruce[1] - tiempos_de_cruce[0]
    return None

T_sim = (periodo_simulacion(t_v, y_r))*24.18884
T_teo = (2 * np.pi)*24.18884

#1 h_barra / E = 24.18884 attosegundos

print(f'2.a) P_teo = {T_teo:.5f}; P_sim = {T_sim:.5f}')

#Punto 2b

alpha = 1/137

def f_prime_larmor(t,f):

    x, y, vx, vy, a = f

    r = np.sqrt(x**2 + y**2)

    ax = -x/np.abs(r**3)
    ay = -y/np.abs(r**3)

    a_m = np.sqrt(ax**2 + ay**2)

    return np.array([vx,vy,ax,ay,a_m])

#Método Runge Kutta (4to orden)
def RK4_stepl(F,y0,t,dt):
    k1 = F(t,y0)
    k2 = F( t+dt/2, y0 + dt*k1/2 )
    k3 = F( t+dt/2, y0 + dt*k2/2  )
    k4 = F( t+dt, y0 + dt*k3  )
    y_siguiente = y0 + dt/6 * (k1+2*k2+2*k3+k4)

    vx,vy = y_siguiente[2],y_siguiente[3]

    v_m = np.sqrt(y_siguiente[2]**2 + y_siguiente[3]**2)
    a2 = y_siguiente[4] **2

    if t > 0:
        v_m_nueva = np.sqrt(max((v_m)**2 - ((4/3)*(alpha**3) * a2**2 * dt),0))

        if v_m_nueva > 0:

           v_v_unitario = np.array([vx,vy])/v_m
           v_v_actualizado = v_v_unitario * v_m_nueva

           y_siguiente[2], y_siguiente[3] = v_v_actualizado

    return y_siguiente

def runge_kuttal(F,y_0,ts,dt):

    t_v = [ts[0]]
    y_v = [y_0]

    #El while establece una condicion de que se itere siempe y cuando la distancia al centro del origen sea mayor a 0.0000001 radios atómicos
    while np.sqrt(y_v[-1][0]**2 + y_v[-1][1]**2) > 0.0000001**2 and t_v[-1] < ts[1]:
        t = t_v[-1]
        y_next = RK4_stepl(F, y_v[-1], t, dt)
        t_v.append(t + dt)
        y_v.append(y_next)

    return np.array(t_v), np.array(y_v)

condiciones_iniciales_l =np.array([x0, y0, vx0, vy0, 0])
t_sl = (0,25.7999)
dt_l = 0.0001

t_vl, y_vl = runge_kuttal(f_prime_larmor, condiciones_iniciales_l, t_sl, dt_l)

xl, yl, vxl, vyl, a_ml = y_vl[:, 0], y_vl[:, 1], y_vl[:, 2], y_vl[:, 3], y_vl[:,4]

# Calcular energía en cada instante
rl = np.sqrt(xl**2 + yl**2)
v2l = np.sqrt(vxl**2 + vyl**2)
Kl = 0.5 * v2l  # Energía cinética
Ul = -1 / rl    # Energía potencial
El = Kl + Ul     # Energía total



# Graficar la energía total, cinética y radio vs tiempo
fig, axs = plt.subplots(3, 1, figsize=(8,10))
axs[0].plot(t_vl, El, label='Energía Total')
axs[1].plot(t_vl, Kl, label='Energía Cinética', color='orange')
axs[2].plot(t_vl, rl, label='Radio', color='green')

for ax, title in zip(axs, ["Energía Total", "Energía Cinética", "Radio"]):
    ax.set_xlabel("Tiempo (unidades atómicas)")
    ax.set_ylabel(title)
    ax.legend()
    ax.grid()

plt.savefig("Talleres/Taller_3/2.b.diagnostics.pdf")
plt.close()


paso_animacion = 1000
xl_anim, yl_anim = xl[::paso_animacion], yl[::paso_animacion]

fig, ax = plt.subplots()
ax.set_xlim(min(xl_anim[-5000:]) - 0.01, max(xl_anim[-5000:]) + 0.01)
ax.set_ylim(min(yl_anim[-5000:]) - 0.01, max(yl_anim[-5000:]) + 0.01)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Órbita del Electrón")

proton, = ax.plot(0, 0, 'ro', markersize=8, label="Protón")
punto, = ax.plot([], [], 'bo', markersize=5, label="Electrón")
trayectoria, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)

ax.legend()

def init():
    punto.set_data([], [])
    trayectoria.set_data([], [])
    return punto, trayectoria

def update(frame):
    punto.set_data([xl_anim[frame]], [yl_anim[frame]])
    trayectoria.set_data(xl_anim[:frame+1], yl_anim[:frame+1])
    return punto, trayectoria

ani = animation.FuncAnimation(fig, update, frames=len(xl_anim), init_func=init, interval=1)

ani.save("Talleres/Taller_3/orbita_electron.mp4", writer=animation.FFMpegWriter(fps=30))



tiempo_de_caida = t_vl[-1] * 24.188
print(f'2.b El electrón tiene un tiempo de caida de {tiempo_de_caida:5f} attosegundos')

#Punto4 (niveles de energía)
def ec_schordinger (x,y,E):
    #y es el vector = [f,f´]
    f,df= y
    ddf = (x**2-2*E)*f
    return [df,ddf]

def evento_convergencia(x, y, E, threshold=12):
    # y[0] es f(x)
    hipotenusa = np.sqrt(y[0]**2 + y[1]**2)
    return threshold - hipotenusa

def energía_par(E, x_max=6.0):
    y0 = [0.3, 0.0]
    x_span = (0, x_max)
    
    ev_func = lambda x, y: evento_convergencia(x, y, E)
    ev_func.terminal = True
    ev_func.direction = -1

    sol = solve_ivp(
        lambda x, y: ec_schordinger(x, y, E),
        x_span,
        y0,
        rtol=1e-7, 
        atol=1e-9,
        events=lambda x, y: evento_convergencia(x, y, E)   
    )
    
    f_at_end = sol.y[0][-1] 
    event_triggered = (len(sol.t_events[0]) > 0)
    return sol, event_triggered

def energia_impar(E, x_max=6.0):
    y0 = [0.0, 1.0]  # f(0)=1, f'(0)=0 (pares)
    x_span = (0, x_max)
    
    ev_func = lambda x, y: evento_convergencia(x, y, E)
    ev_func.terminal = True
    ev_func.direction = -1

    sol = solve_ivp(
        lambda x, y: ec_schordinger(x, y, E),
        x_span,
        y0,
        rtol=1e-7, 
        atol=1e-9,
        events=lambda x, y: evento_convergencia(x, y, E)   
    )
    
    f_at_end = sol.y[0][-1] 
    event_triggered = (len(sol.t_events[0]) > 0)
    return sol, event_triggered

energias_A = np.arange(0.0, 9.52, 0.1)
candidatos_A = []  # Aquí se guardarán las energías donde la solución converge

for E in energias_A:
    sol, event_triggered = energia_impar(E)
    if not event_triggered:
        candidatos_A.append(E)

candidatos_A = np.array(candidatos_A)

energies = np.arange(0.0, 10.0, 0.1)
candidatos = []  # Aquí se guardarán las energías donde la solución converge

for E_ in energies:
    sol, event_triggered = energía_par(E_)
    if not event_triggered:
        candidatos.append(E_)

candidatos = np.array(candidatos)

plt.figure(figsize=(6,7))
x=np.linspace(-6,6,200)
plt.plot(x,1/2*x**2,linestyle="--",color="lightgray")

for e in candidatos:
    sol, _ = energía_par(e)
    t_negativo = -sol.t[::-1]
    y_negativo= sol.y[0][::-1] 
    t=np.concatenate((t_negativo, sol.t))
    E= np.concatenate((y_negativo, sol.y[0]))
    E= E+e
    plt.axhline(y=e, color='lightgray')
    plt.plot(t, E)

for r in candidatos_A:
    sol_a,_= energia_impar(r)
    t_n = -sol_a.t[::-1]
    y_n= -sol_a.y[0][::-1]
    t_A= np.concatenate((t_n, sol_a.t))
    E_An= np.concatenate((y_n,sol_a.y[0]))
    E_An= E_An+r
    plt.axhline(y=r, color='lightgray')
    plt.plot(t_A,E_An)


plt.ylim(0,10)
plt.xlim(-6,6)
plt.ylabel("Energía")
ruta_guardar=  ('Talleres/Taller_3/4.pdf')
plt.savefig(ruta_guardar, format="pdf", dpi=300, bbox_inches="tight")
plt.clf()

#parte b---------------------------------------------------------------------------------------------------

#Punto 1
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
ruta_guardar=  ('Talleres/Taller_3/1.png')
plt.savefig(ruta_guardar,dpi=300,bbox_inches="tight")
plt.clf()

#Punto 2
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
ani.save(r"Talleres\Taller_3\2.mp4", writer=writer)

plt.close(fig)
