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

plt.plot(t, sol_ah[:, 0], label='x(t) ah')
plt.plot(t, sol_ah[:, 1], label="y(t) ah")
plt.plot(t, sol_h[:, 0], label='x(t) h')
plt.plot(t, sol_h[:, 1], label="y(t) h")
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.xlabel('t')
plt.grid()
plt.savefig("q1_1.png", dpi=500)
plt.close()

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
plt.savefig('q1_2.png', dpi=500)
plt.close()

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
plt.savefig('q1_3.png', dpi=500)
plt.close()
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
plt.savefig('q1_4.png', dpi=500)
plt.close()

# Code for question 1.5 
y0 = [1,0]
p=[2,4,6,8,10]

for P in p:
    solm2_h=odeint(model2, y0, t, args=(m,k,P))
    plt.plot(t, solm2_h[:, 0], label='$x(t), \, p=$'+str(P))
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.title('Comparison of different values of $p$')
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.grid()
plt.savefig('q1_5_motion.png', dpi=500)
plt.close()

X2 = np.linspace(-1.223, 1.223, 500)

for P in p:
     plt.plot(X2 ,(1 / P) * k * X2 ** P, label='$p=$' + str(P))

plt.title("$V(x)$ for different p values")
plt.xlabel("$x$")
plt.ylabel("$V(x)$")
plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.grid()
plt.savefig('q1_5_potential.png', dpi=500)
plt.close()

#code for q2.1

#we now add an external force, F_0*np.sin(o*t)
#new EoM: x''=(F_0/m)*np.sin(o*t)-(k/m)*x**(p-1)
#we keep p=2
#we keep the same m, k and t

O = [1/4,1/2,4] #we set omega to a different value from the natural frequency of the system
t = np.linspace(0, 100, 200001)
def model3(u, t, m, k, p, o, F_0):
    x, y = u
    dzdt = [y, (F_0/m)*np.sin(o*t)-(k/m)*x**(p-1)]  # System of first order differential equations [y, y'] Where y=x'
    return dzdt

y0 = [1, 0]
F = [1,20,50,200]

for o in O:
    for f in F:
        solm3_F = odeint(model3, y0, t, args=(m, k, p1, o, f))
        plt.plot(t, solm3_F[:, 0], label='$x(t), \, F=$' + str(f))

    plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
    plt.title('The Gorilla effect for $\omega=$' + str(o) + '$\, p=2$')
    plt.ylabel('$x(t)$')
    plt.xlabel('$t$')
    plt.grid()
    plt.savefig('q2_1_omega'+ str(o)+'.png', dpi=500)
    plt.close()

#code for q2.2

# now setting p = 4

for o in O:
    for f in F:
        solm3_F = odeint(model3, y0, t, args=(m, k, p2, o, f))
        plt.plot(t, solm3_F[:, 0], label='$x(t), \, F=$' + str(f))

    plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
    plt.ylabel('$x(t)$')
    plt.xlabel('$t$')
    plt.title('The Gorilla effect for $\omega=$' + str(o) + '$\, p=4$')
    plt.grid()
    plt.savefig('q2_2_omega'+ str(o)+'.png', dpi=500)
    plt.close()

# code for q2.3
O = [0.5, 1.1, 2]
f = 0.7
for o in O:
    solm3_F = odeint(model3, y0, t, args=(m, k, p1, o, f))
    plt.plot(t, solm3_F[:, 0], label='$x(t), \, \omega=$' + str(o))

plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.ylabel('$x(t)$')
plt.xlabel('$t$')
plt.grid()
plt.title('Demonstration of beats')
plt.savefig('q2_3.png', dpi=500)
plt.close()

# code for q2.4
O = [1/10, 1/5, 0.9, 5, 10]
for o in O:
    solm3_F = odeint(model3, y0, t, args=(m, k, p1, o, f))
    plt.plot(t, solm3_F[:, 0], label='$x(t), \, \omega=$' + str(o))

plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.ylabel('$x(t)$')
plt.xlabel('$t$')
plt.grid()
plt.title('Comparison of frequencies for $p=2$')
plt.savefig('q2_4_p2.png', dpi=500)
plt.close()

for o in O:
    solm3_F = odeint(model3, y0, t, args=(m, k, p2, o, 1.6))
    plt.plot(t, solm3_F[:, 0], label='$x(t), \, \omega=$' + str(o))

plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
plt.ylabel('$x(t)$')
plt.xlabel('$t$')
plt.grid()
plt.title('Comparison of frequencies for $p=4$')
plt.savefig('q2_4_p4.png', dpi=500)
plt.close()

# code for q3.1

def model_friction(u, t, m, k, p, o, F_0, b):
    x, y = u
    dzdt = [y, (F_0/m)*np.sin(o*t)-(k/m)*x**(p-1)- b*y]  # System of first order differential equations [y, y'] Where y=x'
    return dzdt

B = [0.01, 1, 100]
O = np.linspace(0.1,10,1000)
# Note: this loop takes about 2 hours to run
for b in B:
    amplitudes = [np.amax(odeint(model_friction, y0, np.linspace(0, 20000 * np.pi / o + 1, 100000), args=(m, k, p1, o, f, b))[:,0][90000:]) for o in O]
    plt.plot(O, amplitudes, label='$A(\omega), \, b = $' + str(b))
    plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
    plt.ylabel('$A(\omega)$')
    plt.xlabel('$\omega$')
    plt.grid()
    plt.title('The relationship between amplitude and frequency for $b = $' + str(b))
    plt.savefig('q3_1b_' + str(b) + '.png', dpi=500)
    plt.close()
# code for q3.2
B = [0.01, 2, 100]
print(o)
o = 10
f = 0.01
for p in [p1,p2]:
    for b in B:
        sol_friction = odeint(model_friction, y0, t, args=(m, k, p, o, f, b))
        plt.plot(t, sol_friction[:, 0], label='$x(t), \, b = $' + str(b))

    plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
    plt.ylabel('$x(t)$')
    plt.xlabel('$\omega$')
    plt.grid()
    plt.title('$x(t)$ for different values of $b, \, F_0=$'+str(f) + '$, \,p= $ ' + str(p))
    plt.savefig('q3_2_p_'+ str(p) +'.png', dpi=500)
    plt.close()

# code for 3.3
F = [1, 10]
for p in [p1, p2]:
    for f in F:
        for b in B:
            sol_friction = odeint(model_friction, y0, t, args=(m, k, p, o, f, b))
            plt.plot(t, sol_friction[:, 0], label='$x(t), \, b = $' + str(b))

        plt.legend(loc='upper center', bbox_to_anchor=(1, 1),  shadow=True, ncol=1)
        plt.ylabel('$x(t)$')
        plt.xlabel('$\omega$')
        plt.grid()
        plt.title('$x(t)$ for different values of $b, \, F_0=$'+str(f) + '$, \, p=$' + str(p))
        plt.savefig('q3_3_f_' +str(f)+'p_' + str(p) + '.png', dpi=500)
        plt.close()


