import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Example code for Model 1 --------------------------------------

# Model1 defines the Equations Of Motion (hereafter: EoM) for a potential V(x) = (1/2)kx^2(1-(2/3)ax). For such a potential
# and no external forces (F_ext(x,t)=0), the EoM is given by Newton's second law (remember F_k(x) = -dV/dx)

# EoM  :  mx''= -kx(1-ax)
# When ax<<1  : Harmonic Motion
# When x⟶1/a : Anharmonic Motion


t = np.linspace(0, 20, 201)  # Time

m = 1  # keeping k=m=1 for simplicity
k = 1
a1 = 0.5
a2 = 0.0001


# To solve the EoM, we need to convert the EoM to a system of first order Diffrential Equations, i.e. introduce y=x'
def model1(z, t, m, k, a):
    # z: contains the initial conditions [x(0), x'(0)]
    # t: the time coordinate
    # m, k : mass and spring constant respectively
    # a : perturbation parameter that introduces non-linearity

    x, y = z
    dzdt = [y, -(k / m) * x * (1 - a * x)]  # System of first order differential equations [y, y'] Where y=x'
    return dzdt

y0 = [1.8, 0]  # Initial conditions [x(0), x'(0)]

# Solves the system by integrating y(t) and y'(t) to obtain x(t) and x'(t) (since y(t)=x'(t)).
# Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html -> check the bottom of this
# page for an example

#code for q1.1
sol_ah = odeint(model1, y0, t, args=(m, k, a1))
sol_h = odeint(model1, y0, t, args=(m, k, a2))

plt.plot(t, sol_ah[:, 0], 'b', label='x(t) ah')
plt.plot(t, sol_ah[:, 1], 'g', label="y(t) ah")
plt.plot(t, sol_h[:, 0], 'r', label='x(t) h')
plt.plot(t, sol_h[:, 1], 'm', label="y(t) h")
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()


Amplitudes = [0.5,1.25,1.85,1.95]   #Different Initial conditions for x(0)

x = np.zeros((len(t),len(Amplitudes)))   #Create arrays to store results for x(t) & x'(t) for all the different amplitudes
xdot = np.zeros((len(t),len(Amplitudes)))



#code for q1.2
for i in range(len(Amplitudes)):                     #Solves EoM for all the different amplitudes
    y1=[Amplitudes[i],0]                             # initial conditions [x(0), x'(0)]
    sol_ah_amp = odeint(model1, y1, t, args=(m,k,a1))
    x[:,i] = sol_ah_amp[:,0]
    xdot[:,i] = sol_ah_amp[:,1]

for i in range(len(Amplitudes)):
    plt.plot(t, x[:, i], label='x(0)={}'.format(Amplitudes[i]))
    plt.xlabel('time')
    plt.ylabel('x(t)')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()


#code for q1.3
# plots the potentials for harmonic motion, V(x) = (1/2)kx^2, and anharmonic motion V(x) = (1/2)kx^2(1-(2/3)ax) for a=0.1

a3 = 0.1

X = np.linspace(-10, 10, 500)
X1 = np.linspace(-10, 16, 1500)

harmonicPotential = (1 / 2) * k * X ** 2
AnharmonicPotential = ((1 / 2) * k * X1 ** 2) * (1 - (2 / 3) * a3 * X1)

plt.plot(X, harmonicPotential, label='Harmonic')
plt.plot(X1, AnharmonicPotential, label='Anharmonic', ls='--')
plt.plot(1 / a3, 0, 'rx', label='x=1/a')
plt.legend(loc='upper center', bbox_to_anchor=(1.35, 1), shadow=True, ncol=1)
plt.title("V(x) for a=0.1")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid()
plt.show()

#code for q1.4

#Model 2 has a different EoM due to the change in potential
#V(x)=(1/p)*k*x**p
#EoM: x''=-(k/m)*x**(p-1)
#p=2n
#Apparently only p=2 produces harmonic motion, crazy
#We're keeping the values for m, k, and we'll stay with the same t-range


p1=2
p2=4
def model2(r, t, m, k, p):
    #r: contains the initial conditions [x(0), x'(0)]
    #t: time, obviously
    #m
    #k
    #p: power, has to be even
    x, y = r
    drdt = [y, -(k/m)*x**(p-1)] #y=x'
    return drdt

y0=[1.5, 0] #initial conditions [x(0), x'(0)]

solm2_h=odeint(model2, y0, t, args=(m,k,p1))
solm2_ah=odeint(model2, y0, t, args=(m,k,p2))

plt.plot(t, solm2_h[:, 0], 'r', label='x(t) h')
plt.plot(t, solm2_h[:, 1], 'm', label="y(t) h")

plt.plot(t, solm2_ah[:, 0], 'y', label='x(t) ah')
plt.plot(t, solm2_ah[:, 1], 'k', label="y(t) ah")
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

p=[2,4,6,8,10]

x1 = np.zeros((len(t),len(p)))   #Create arrays to store results for x(t) & x'(t) for all the different amplitudes
x1dot = np.zeros((len(t),len(p)))

for i in range(len(p)):
    solm2_p = odeint(model2, y0, t, args=(m, k, p[i]))
    x1[:, i] = solm2_p[:, 0]
    x1dot[:, i] = solm2_p[:, 1]

for i in range(len(p)):
    plt.plot(t, x1[:, i], label='p={}'.format(p[i]))
    plt.xlabel('time')
    plt.ylabel('x(t)')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()


X2 = np.linspace(-10, 10, 500)

harmonicPotential1 = (1/2)*k*X2**2
anharmonicPotential1 = np.zeros((len(t),len(X2))) #maybe this should be 3d?

for i in range(len(p)):
     np.append(anharmonicPotential1[i], (1 / p[i]) * k * X2 ** p[i])

plt.plot(X2, harmonicPotential1, label='Harmonic')

for i in range(len(p)):
    plt.plot(X2, anharmonicPotential1[i], label='Anharmonic', ls='--')
plt.plot(X2, (1 / 4 * k * X2 ** 4), label='Anharmonic 4', ls='--')
plt.legend(loc='upper center', bbox_to_anchor=(1.35, 1), shadow=True, ncol=1)
plt.title("V(x) for different p values")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.grid()
plt.show()

#code for q2.1

#we now add an external force, F_0*np.sin(o*t)
#new EoM: x''=(F_0/m)*np.sin(o*t)-(k/m)*x**(p-1)
#we keep p=2
#we keep the same m, k and t

o1 = 2
o2 = 3
o3 = 4
def model3(u, t, m, k, p, o, F_0):
    x, y = u
    dzdt = [y, (F_0/m)*np.sin(o*t)-(k/m)*x**(p-1)]  # System of first order differential equations [y, y'] Where y=x'
    return dzdt

y0 = [1.5, 0]

F1 = 1
F2 = 20
F3 = 50
F4 = 200

solm3_F1o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F1))
solm3_F2o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F2))
solm3_F3o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F3))
solm3_F4o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F4))

plt.plot(t, solm3_F1o1[:, 0], 'y', label='x(t) F1=1, o1')
plt.plot(t, solm3_F2o1[:, 0], 'r', label='x(t) F2=20, o1')
plt.plot(t, solm3_F3o1[:, 0], 'm', label='x(t) F3=50, o1')
plt.plot(t, solm3_F4o1[:, 0], 'b', label='x(t) F4=200, o1')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

solm3_F1o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F1))
solm3_F2o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F2))
solm3_F3o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F3))
solm3_F4o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F4))

plt.plot(t, solm3_F1o2[:, 0], 'y', label='x(t) F1=1, o2')
plt.plot(t, solm3_F2o2[:, 0], 'r', label='x(t) F2=20, o2')
plt.plot(t, solm3_F3o2[:, 0], 'm', label='x(t) F3=50, o2')
plt.plot(t, solm3_F4o2[:, 0], 'b', label='x(t) F4=200, o2')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

solm3_F1o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F1))
solm3_F2o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F2))
solm3_F3o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F3))
solm3_F4o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F4))

plt.plot(t, solm3_F1o3[:, 0], 'y', label='x(t) F1=1, o3')
plt.plot(t, solm3_F2o3[:, 0], 'r', label='x(t) F2=20, o3')
plt.plot(t, solm3_F3o3[:, 0], 'm', label='x(t) F3=50, o3')
plt.plot(t, solm3_F4o3[:, 0], 'b', label='x(t) F4=200, o3')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

#code for q2.2
#same thing but p=4

p1=4

solm3_F1o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F1))
solm3_F2o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F2))
solm3_F3o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F3))
solm3_F4o1 = odeint(model3, y0, t, args=(m, k, p1, o1, F4))

plt.plot(t, solm3_F1o1[:, 0], 'y', label='x(t) F1=1, o1')
plt.plot(t, solm3_F2o1[:, 0], 'r', label='x(t) F2=20, o1')
plt.plot(t, solm3_F3o1[:, 0], 'm', label='x(t) F3=50, o1')
plt.plot(t, solm3_F4o1[:, 0], 'b', label='x(t) F4=200, o1')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

solm3_F1o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F1))
solm3_F2o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F2))
solm3_F3o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F3))
solm3_F4o2 = odeint(model3, y0, t, args=(m, k, p1, o2, F4))

plt.plot(t, solm3_F1o2[:, 0], 'y', label='x(t) F1=1, o2')
plt.plot(t, solm3_F2o2[:, 0], 'r', label='x(t) F2=20, o2')
plt.plot(t, solm3_F3o2[:, 0], 'm', label='x(t) F3=50, o2')
plt.plot(t, solm3_F4o2[:, 0], 'b', label='x(t) F4=200, o2')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()

solm3_F1o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F1))
solm3_F2o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F2))
solm3_F3o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F3))
solm3_F4o3 = odeint(model3, y0, t, args=(m, k, p1, o3, F4))

plt.plot(t, solm3_F1o3[:, 0], 'y', label='x(t) F1=1, o3')
plt.plot(t, solm3_F2o3[:, 0], 'r', label='x(t) F2=20, o3')
plt.plot(t, solm3_F3o3[:, 0], 'm', label='x(t) F3=50, o3')
plt.plot(t, solm3_F4o3[:, 0], 'b', label='x(t) F4=200, o3')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.show()