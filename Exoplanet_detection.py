import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



def astro(radio,xposit):
    
    '''
    devuelve una matriz con todos sus elementos igual a cero
    excepto un circulo de radio (radio), centrado en xposit
    '''
    height, width = 200, 200
    img = np.zeros((height, width), np.uint8)
    img[:, :] = [255]
    

    row, col = int(height/2), int(width/2)+int(xposit)
   

    astro=cv2.circle(img,(col, row), radio, (0), -1)

    return astro 

def seno_creciente(t):
    
    '''
    devuelve la funcion seno en las regiones que el seno es creciente
    y 1 en las regiones que es decreciente. el argumento t es un array
    '''
    
    z=[]

    for val in t:
        if 0<math.cos(val):
                
            newval=math.sin(val)
           
        else:
            newval=1
        z.append(newval)
    return np.array(z) 


def observador(posicion):
    '''
    devuelve la diferencia entre las iamgenes de los astros de Radio R y r
    dada la posicion del planeta
    '''
    
    plt.axis('off')
    diff=-astro(R,0)+astro(r,posicion)
    diff=cv2.blur(diff, (3, 5))
    imagen=plt.imshow(diff,cmap ="binary_r")
    
    return imagen,diff

def signal(x):
    
    '''
    devuelbe la señal intensidad del flujo luminoso (f)
    leida como el valor medio de la imagene segun la posicion x del planeta
    f es un array
    '''
    f=[]
    
    for i in x:
        xposit=int(i)
        sol=cv2.blur(astro(R,0), (57,57))
        
        diff=-sol+astro(r,xposit)
        
        diff=cv2.blur(diff, (5,5))
        
        flujo=diff.mean()
        
        f.append(flujo)
     
    return f

    M=50

R=55
r=5
D=700
G= 1

observador(35)[0]


v=np.sqrt(G*M/D)


tfinal=15500
Nsamples=tfinal//10
t=np.linspace(0,tfinal*np.pi,Nsamples)

decay=1 #*np.exp(-0.0005*t)

x=D*seno_creciente(v*t/D)*decay
    
plt.plot(t,x,label='(planeta tapado)')
#plt.plot(t,D*np.sin(v*t/D)*decay,  '--', label='sin tapar')



plt.axhspan(-R, R, alpha=0.1, color='orange', label='haz de la estrella')
plt.xlabel('tiempo')
plt.ylabel('amplitud orbital corrdenada x')
plt.legend()


plt.show()

ruido = 0.0051 * np.random.normal(size=x.size)

y=signal(x)+ruido



plt.plot(t,y)#,color='black')
plt.xlabel('tiempo')
plt.ylabel('flujo de intensidad luminica')
plt.title('señal obtenida por el observador')
#plt.xlim((16200,16600))
plt.grid()
plt.show()

plt.plot(t,y,"-*")
plt.title('zoom en uno de los pozos')
plt.xlabel('tiempo')
plt.ylabel('flujo de intensidad luminica')

plt.xlim((16200,16600))
plt.grid()
plt.show()


from math import *
def analize(F_notransit, F_transit, t_F, t_T, P):
    #t_F= tiempo de transito (abajo)
    #t_T (arriba)
    k=sqrt((F_notransit - F_transit) / F_notransit)
    b=sqrt((1-k)**2-(t_F/t_T)**2*(1+k)**2) / sqrt(1-(t_F/t_T)**2)
    a=sqrt((1+k)**2-b**2*(1-sin(t_T*np.pi/P)**2)) / sin(t_T*np.pi/P)
    return k, a


an=analize(60.41,59.9,100,250,40000/3)
print('Parametros usados, parametros estimados')
print(r/R,"|",an[0])
print(D/R,"|",an[1])


def hab_zone(L, Teff, r_orbit):
    
    r_orbit=r_orbit/500 #(AU)
    Ts=5700 #K
    ai=2.7619E-5
    bi=3.8095E-9
    ao=1.3786E-4
    bo=1.4286E-9
    ris=0.72
    ros=1.77
    
    ri = (ris - ai*(Teff - Ts) - bi*((Teff - Ts))**2)/sqrt(L)
    ro = (ros - ao*(Teff - Ts) - bo*((Teff - Ts))**2)/sqrt(L)
    
    HZD= (2*r_orbit - ri - ro) /(ro-ri)
    
    return abs(HZD)


if hab_zone(1, 6000, D)<1 and hab_zone(1, 6000, D)>0.1:
    print('planeta en zona habitable')
    
else:
    print('..')


import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
decay=1# *np.exp(-0.000005*t)
x = D * np.sin(v*t/D)*decay
y = D * np.cos(v*t/D)*decay
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

ax.set_xlim((-D-100,D+100))
ax.set_ylim((-D-100,D+100))

particle, = ax.plot([],[], marker='o', markersize=r,color='b')
traj, = ax.plot([],[], alpha=0.5,color='b')


def update(i):
    
    particle.set_data(x[i],y[i])
    particle.set_label('time:'+str(i))
    traj.set_data(x[:i+1],y[:i+1])
    plt.legend(bbox_to_anchor=(.04,1), loc="upper left")
    return particle,traj

ani = animation.FuncAnimation(fig, update,interval=100)

ax.set_aspect('equal', adjustable='box')
plt.axvspan(-R, R, alpha=0.1, color='orange', label='linea estrella-observador')
plt.plot(0,0, 'o', markersize=R,color='orange')
plt.plot(0,D+90, 'v',color='black', label='observador')
plt.xlabel('cordeenada x')
plt.ylabel('cordeenada y')
#plt.axhline(y=.5, xmin=0.5, xmax=0.0013*D, color='black',label='radio orbital $D$')
#plt.xticks([0,15],label='aca va el cero')

ax.annotate('distancia orbital $D$', 
            xy=(0, 0), 
            xytext=(D+10, -10), 
            arrowprops = dict(arrowstyle='|-|',facecolor='black')
           )


ax.annotate('radio estelar $R$', 
            xy=(-R,-200), 
            xytext=(0,-210), 
            arrowprops = dict(arrowstyle='|-|',facecolor='orange')
           )

plt.show()

print('si no se ve nada reiniciar el kernel')


