#Librerias
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar

"Punto 1"

# 1.a

#Los parámetros iniciales estan dados por
g=9.773
v0=10
m=10

def x_max(theta_0, beta):
    theta_0=np.radians(theta_0)
    v_x=v0*np.cos(theta_0)
    v_y=v0*np.sin(theta_0)
    W0=[0, 0, v_x, v_y]
    tvuelo1=2*v0*np.sin(theta_0)/g
    tvuelo2=(0, tvuelo1* 1.5) #definen los intervalos de tiempo y el tiempo de vuelo se multiplica por 1.5 para que la simulación tome el tiempo en que cayó el cuerpo.

    def sistema(t, W, theta_0, beta): #definimos la ecuaciones diferenciales
     x, y, vx, vy=W
     v=np.sqrt((vx**2)+(vy**2)) #ecuación para la velocidad
     a_x=(-beta)*v*vx/m #ecuación de la aceleración en x
     a_y=-g -(beta*v*vy/m) #ecuación de la aceleración en y
     return [vx, vy, a_x, a_y] #se devuelven las derivadas
    
    solucion=solve_ivp(sistema, tvuelo2, W0, args=(theta_0, beta), dense_output=True)
    
    # Encontrar el primer punto donde y cambia de signo (aproximación del impacto)

    for i in range(len(solucion.y[1])-1): #se usa solucion.y para ordenar matricialmente los cambios en el tiempo de x, y, v_x y v_y. La posición solucion.1 es posicion de y
        if solucion.y[1,i]*solucion.y[1,i+1]<0: #y se vuelve negativo
            return (solucion.y[0, i], solucion.y[1, i], solucion.y[2, i], solucion.y[3, i])  #devuelve x del impacto
    return (solucion.y[0,-1], solucion.y[1,-1], solucion.y[2,-1], solucion.y[3,-1])  # Si no hay intersección,se devuelve el último x registrado

def theta_max(beta):
    return minimize_scalar(lambda theta: -x_max(theta, beta)[0], bounds=(0, 90), method='bounded').x
 #el maximo de x_max es el minimo de -x_max
beta=0.1
print(f"1.a) El ángulo óptimo para beta={beta} es {theta_max(beta):.2f} grados")

# 1.b

def deltae(beta):
    x_final, y_final, vxf, vyf = x_max(theta_max(beta), beta)
    vf = np.sqrt((vxf**2)+(vyf**2))
    E0=(1/2)*m*(v0**2)
    Ef=(1/2)*m*(vf**2) #se definen las energías para cada una
    resp=abs(Ef-E0) #evitar negativos
    return resp
theta_0=np.radians(45)
bs=np.logspace(0, 0.3, 50)
bs2=np.logspace(0, 0.3, 50)
energias=[]
thetasmax=[]
for b in bs:
    energias.append(deltae(b))
for b in bs2:
    thetasmax.append(theta_max(b))
plt.figure(figsize=(8, 5))
plt.plot(bs2, thetasmax, marker='x', linestyle='--', color='r')
plt.xscale('log')
plt.xlabel(r" $\beta$ (kg/m)")
plt.ylabel(r"$\theta$ (grados)")
plt.grid()
plt.savefig("Talleres/Taller_3/1.a.pdf")

plt.figure(figsize=(8, 5))
plt.plot(bs, energias, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel(r" $\beta$ (kg/m)")
plt.ylabel(r"$\Delta E$ (J)")
plt.grid()
plt.savefig("Talleres/Taller_3/1.b.pdf")
print("1.b) La energía perdida cuando el coeficiente de fricción es 0 es: ", deltae(0))

"Punto 2"

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

@njit
def f_prime_larmor(t,f):

    x, y, vx, vy, a = f

    r = np.sqrt(x**2 + y**2)

    ax = -x/(r**3)
    ay = -y/(r**3)

    a_m = np.sqrt(ax**2 + ay**2)

    return np.array([vx,vy,ax,ay,a_m], dtype=np.float64)

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
        v_m_nueva = np.sqrt(max((v_m)**2 - ((4/3)*(alpha**3) * a2 * dt),0))

        if v_m_nueva > 0:

           v_v_unitario = np.array([vx,vy], dtype=np.float64)/v_m
           v_v_actualizado = v_v_unitario * v_m_nueva

           y_siguiente[2], y_siguiente[3] = v_v_actualizado[0],v_v_actualizado[1]

    return y_siguiente

def runge_kuttal(F,y_0,ts,dt):

    max_pasos = 10000

    t_v = np.empty(max_pasos, dtype=np.float64)
    y_v = np.empty((max_pasos, y_0.shape[0]), dtype=np.float64)

    t_v[0] = ts[0]
    y_v[0, :] = y_0.copy()

    i = 0
    u_indice = None
    r_a = 1

    while True:
        t = t_v[i]
        y_next = RK4_stepl(F, y_v[i], t, dt)

        r = np.sqrt(y_next[0]**2 + y_next[1]**2)  # Distancia al origen

        if r > 0.01 and r < r_a:
            u_indice = i
        if r < 0.01 or t >= ts[1]:  # Condición de salida
            break
        
        i += 1

        t_v[i] = t + dt
        y_v[i, :] = y_next

        r_a = r

    return t_v[:i+1], y_v[:i+1],u_indice

condiciones_iniciales_l =np.array([x0, y0, vx0, vy0, 0.0])
t_sl = (0,1000)
dt_l = 0.1

t_vl, y_vl,u_indice = runge_kuttal(f_prime_larmor, condiciones_iniciales_l, t_sl, dt_l)

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

plt.savefig("Talleres/Taller_3/2b.diagnostics.pdf")  # Guardar la gráfica de diagnósticos
plt.close()

paso_animacion = 3
xl_anim, yl_anim = xl[::paso_animacion], yl[::paso_animacion]

fig, ax = plt.subplots()
ax.set_xlim(min(xl_anim[-5000:]) - 0.01, max(xl_anim[-5000:]) + 0.01)
ax.set_ylim(min(yl_anim[-5000:]) - 0.01, max(yl_anim[-5000:]) + 0.01)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Órbita del Electrón")

proton, = ax.plot(0, 0, 'ro', markersize=8, label="Protón")
puntol, = ax.plot([], [], 'bo', markersize=5, label="Electrón")
trayectoria, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)

ax.legend()

def init():
    puntol.set_data([], [])
    trayectoria.set_data([], [])
    return puntol, trayectoria

def update2(frame):
    puntol.set_data([xl_anim[frame]], [yl_anim[frame]])
    trayectoria.set_data(xl_anim[:frame], yl_anim[:frame])
    return puntol, trayectoria

ani = animation.FuncAnimation(fig, update2, frames=len(xl_anim), init_func=init, interval=1)

ani.save("Talleres/Taller_3/orbita_electron.mp4", writer=animation.FFMpegWriter(fps=30))



tiempo_de_caida = t_vl[u_indice] * 24.188
print(f'2.b El electrón tiene un tiempo de caida de {tiempo_de_caida:5f} attosegundos')


"Punto 3"
#3.a

# Configuración de estilo
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["figure.figsize"] = (14, 5)

# Parámetros
mu = 39.4234021  # Au^3/año^2
alpha_s = 1.09778201e-2  # Au^2
t_span = (0.0, 1.5)  # Intervalo de tiempo

# Función de ecuaciones diferenciales
def ecu_dif(t, Y, mu, alpha_s):
    x, y, vx, vy = Y
    r_vec = np.array([x, y])
    r = np.linalg.norm(r_vec)
    r_unit = r_vec / r

    a = -(mu / r**2) * ((1 + alpha_s / r**2)) * r_unit

    return np.array([vx, vy, a[0], a[1]])

# Condiciones iniciales
alpha_m = 0.38709893  # Au
e = 0.20563069  # Excentricidad
x_0 = alpha_m * (1 + e)
y_0 = 0
vx_0 = 0
vy_0 = np.sqrt((mu / alpha_m) * (1 - e) / (1 + e))
Y0 = np.array([x_0, y_0, vx_0, vy_0])

# Resolver sistema de ecuaciones diferenciales
sol = solve_ivp(ecu_dif, t_span, Y0, args=(mu, alpha_s), max_step=0.01, dense_output=True)

# Interpolación de la solución
t_dense = np.linspace(sol.t[0], sol.t[-2], 300)
Y_dense = sol.sol(t_dense)
x_dense, y_dense, vx_dense, vy_dense = Y_dense

# Configuración de la figura
fig, ax = plt.subplots()
ax.set_xlim(x_dense.min() - 0.15, x_dense.max() + 0.15)
ax.set_ylim(y_dense.min() - 0.15, y_dense.max() + 0.15)

# Inicialización de elementos animados
punto = ax.scatter(*Y0[:2])
linea, = ax.plot([], [], label="Órbita de Mercurio")

# Función de actualización para la animación
def animate(i):
    punto.set_offsets(Y_dense[:2, i])
    if i > 1:
        linea.set_data(x_dense[:i+1], y_dense[:i+1])
    return punto, linea

# Crear la animación
anim = animation.FuncAnimation(fig, animate, frames=range(0, len(t_dense), 1), interval=10)


# Guardar la animación
writer = animation.FFMpegWriter(fps=60)
anim.save(r"Talleres\Taller_3\3.a.mp4", writer=writer)

plt.close(fig)
plt.clf()

#Punto b

t_span_2 = (0., 10.)
alpha_s = 1.09778201e-8 

def calcular_trayectoria(t_span, Y, mu, alpha_s):
    x, y, vx, vy = Y
    return x * vx + y * vy  # Vector

calcular_trayectoria.direction = 0

# Resolver sistema de ecuaciones diferenciales
sol_b = solve_ivp(ecu_dif, t_span_2, Y0, args=(mu, alpha_s), events=calcular_trayectoria, max_step=0.001)

Y_2 = sol_b.y_events[0]

# Afelio
def aphelios(Y_2):
    num_aphelios = len(Y_2)
    aphelios = np.zeros(num_aphelios)

    # Calcular los ángulos de los afelios
    for p in range(num_aphelios):
        aphelios[p] = np.arctan2(Y_2[p][1], Y_2[p][0])

    # Ajustar los ángulos
    for r in range(len(aphelios)):
        ang = aphelios[r]
        dif_pi = np.abs(np.pi - np.abs(ang)) # calcula la diferencia entre el valor absoluto de ang y π
        dif_2pi = np.abs(2 * np.pi - np.abs(ang))  #calcula la diferencia entre el valor absoluto de ang y 2π.
        dif0 = np.abs(ang) #mide la distancia de ang al valor cero.

        if (dif_pi < dif_2pi) and (dif_pi < dif0):
            ang = np.abs(ang) - np.pi
        elif (dif_2pi < dif_pi) and (dif_2pi < dif0):
            ang = np.abs(ang) - 2 * np.pi
        
        aphelios[r] = np.abs(ang)

    return aphelios

aphelios=aphelios(Y_2)

t_aphelios = sol_b.t_events[0] #Tiempos que ocurren los afelios

ang_aphelios = aphelios * (180*3600)/np.pi

tasa_precision = (ang_aphelios[20]-ang_aphelios[6])/(t_aphelios[20]-t_aphelios[6]) * 100 #Afelios bien separados por el tiempo
coef, cov = np.polyfit(t_aphelios, ang_aphelios, 1, cov=True)
incertidumbre = np.sqrt(cov[0, 0]) * 100  # Incertidumbre de la pendiente

#Gráfica

mercury_color = (0.75, 0.50, 0.40)  # Gris metálico con un leve tono marrón
plt.scatter(t_aphelios, ang_aphelios, color=mercury_color, 
            label=rf"Pendiente = {tasa_precision:.4f} $\pm$ {incertidumbre:.4f} $arcsec/siglo$")


plt.xlabel("años")
plt.ylabel(rf"$arcsec$")
plt.title("Presición anómala de Mercurio")
plt.legend()

ruta_guardar_pdf =  ('3_b.pdf')
plt.savefig(ruta_guardar_pdf,format="pdf", bbox_inches='tight')

plt.clf()

"Punto 4"

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
