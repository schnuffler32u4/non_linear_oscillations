# Method chosen
## Solving of equations
In order to minimize the variation of the results, we opted to use the same way of solving the differential equation as in the code provided on Brightspace.
The equations in the solver were modified accordingly, in order to allow for the incorporation of all the required conditions.
Namely, to create the newest implementation according to the equation 
$$
F_{k}(x) = - \frac{dV_{k}}{dx} = -kx^{p-1}
$$
we have to write 
$$
\ddot{x} = - \frac{k}{m} x^{p-1}
$$
Meaning that the vector that is used will be defined as 
$$
z = (x,y)
$$
with, $y =\dot{x}$. This means that we can write $\dot{z}$ as 
$$
\dot{z} = \left(y, -\frac{k}{m} x^{p-1} \right),
$$
which can be used to solve the equations using `scipy.integrate.odeint()`. $p,m,k$ are all entered as parameters of the function to allow for them to be easily changed.

To introduce the time varying force only a small modification is needed, namely:
$$
\dot{z} = (y, F_{0}\sin(\omega t) - kx^{p-1} ).
$$
And lastly, to introduce the equation corresponding to friction proportional to the first derivative of position:
$$
\dot{z} = (y, F_{0}\sin(\omega t) - kx^{p-1}- b y).
$$
## Incorporation of recursive plots
For loops, with defined arrays for all of the required values to use for force and frequency were defined whenever multiple needed to be plotted at the same time to simplify the process. 

## Values chosen
To simplify the code, simple values were chosen for the required constants. This means that $m = 1, k=1 \rightarrow \omega_{0}=1$.

When a question required a value for the force similar to that of $F_{k}$, as $F_{k}$ is not a constant function of position, as an estimate, half of the force at maximum amplitude was used. 

The plot containing both hamornic and aharmonic motion is present in figure x.

