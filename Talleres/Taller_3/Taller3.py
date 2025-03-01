#Librerias
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from mpl_toolkits.mplot3d import Axes3D

#Parte a--------------------------------------------------------------------------------------------------



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
