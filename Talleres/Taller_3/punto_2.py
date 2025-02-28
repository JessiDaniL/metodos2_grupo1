import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Fuerza de Coulbom en unidades atómicas.
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


#Solucion

x_r,y_r,vx_r,vy_r = y_v[:,0],y_v[:,1],y_v[:,2],y_v[:,3]

#Para la animación se utilizo el siguiente código


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
ani.save("2.a.simulation.mp4", writer="ffmpeg", fps=30)

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

    #El while establece una condicion de que se itere siempe y cuando la distancia al centro del origen sea mayor a 0.01 radios atómicos
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

plt.savefig("2.b.diagnostics.pdf")  # Guardar la gráfica de diagnósticos
plt.close()

#Este código se usó para realizar la animación

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

ani.save("orbita_electron.mp4", writer=animation.FFMpegWriter(fps=30))



tiempo_de_caida = t_vl[-1] * 24.188
print(f'2.b El electrón tiene un tiempo de caida de {tiempo_de_caida:5f} attosegundos')