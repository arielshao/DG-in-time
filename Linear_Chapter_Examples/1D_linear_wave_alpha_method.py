from fenics import *
import numpy as np 
import matplotlib.pyplot as plt
import timeit


# We consider the one-dimensional wave-type equation
# u_tt+2*xi*u_t+xi^2*u-u_xx=f in (0,1)x(0,T],
# u(0, t) = u(1,t)=0 for all  t in (0,T], 
# u(x,0)=u_0 and  u_t(x,0)=u_1.

# We set the parameter xi=1 and let u_0, u_1 and f be chosen such that the exact solution is 
# u=sin(sqrt{2}*pi*t)*sin(pi*x).
# That is, u_0=0, u_1= sqrt{2}*pi*sin(pi x), 
# and f=[(-pi^2+xi^2)*sin(\sqrt{2}*pi *t)+2*sqrt{2}*xi*pi *cos(sqrt{2}*pi*t)]*sin(pi* x).



#-----------------------------------------------------------------------------
# Spatial discretization and time-stepping  parameters


nx    = 16            # Number of subintervals in the spatial domain  
tmax    = 1           # Max time 
xmax  = 1             # Max point of x domain 
h= xmax/nx            # calculate the uniform mesh-size 
dt= h                 # time step size 
Nsteps=int(tmax/dt)   # initialize time axis
xi=Constant(1.0)
print('the spatial discretization parameter is', h, 'the temporal discretization parameter is', dt)

# Create mesh and define function space
mesh = UnitIntervalMesh(nx)

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant (0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)



# Define function space for diplacement, velocity and acceleration
V = FunctionSpace(mesh, "CG", 2)

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")

# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)


# Define boundary conditions 
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0), boundary)

# Exact solutions and rhs 
u_exact= Expression('sin(sqrt(2)*pi*t)*sin(pi*x[0])',  degree=10, t=0)
v_exact= Expression("sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(pi*x[0])", degree=10,t=0)
f = Expression('sin(sqrt(2)*pi*t)*sin(pi*x[0])-pow(pi,2)*sin(sqrt(2)*pi*t)*sin(pi*x[0])+2*sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(pi*x[0])',degree=10, t=0)


# Mass form
def m(u, u_):
    return dot(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return dot(grad(u), grad(u_))*dx

# Work of external forces
def Wext(u_):
    return dot(u_, f)*dx 

# Initial conditions
u_old=interpolate(u_exact, V)
v_old=interpolate(v_exact, V)

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old


# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()



def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
res = m(avg(a_old, a_new, alpha_m), u_) + 2*xi*m(avg(v_old, v_new, alpha_f), u_) + xi**2*m(avg(u_old, du, alpha_f), u_)+ k(avg(u_old, du, alpha_f), u_) - Wext(u_)
a_form = lhs(res)
L_form = rhs(res)


# Define solver for reusing factorization
K, res = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True


# Time-stepping
time = np.linspace(0, tmax, Nsteps+1)
tic=timeit.default_timer()
for (i, dt) in enumerate(np.diff(time)):

    t = time[i+1]
    print("Time: ", t)

    # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
    f.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    solver.solve(K, u.vector(), res)


    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Update the exact solutions 
    u_exact.t=t 
    v_exact.t=t 
    f.t = t
    error1=errornorm(u_exact, u, 'L2')
    print('The L2 error ||u-u_DG|| is', error1)
    error2=errornorm( v_exact, v_old, 'L2')
    print('The L2 error ||dot u- dot u_DG|| is', error2 )
    error=error1+error2
    print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )

toc=timeit.default_timer()
print(toc-tic, 'sec Elapsed')

# Generalized alpha method wiht alpha_f=alpha_m=0 --> Newmark Beta Method, 
# T=1, xi=1, h=k, CG p in space


# |  p   |     k     | L2 norm (u)   |  rate   | L2 norm (v) |  rate  |  L2 norm  |  rate  |
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|:---------:|:------:|
# |  2   |5.00e-1    |  6.6797e-1   |   ----   |  1.2325e-0  |  ----  | 1.9004e-0 |  ----  | 
# |      |2.50e-1    |  1.8100e-1   |  1.8838  |  5.1116e-1  | 1.2697 | 6.9216e-1 | 1.4571 |                                  
# |      |1.25e-1    |  4.4837e-2   |  2.0132  |  1.4317e-1  | 1.8360 | 1.8801e-1 | 1,8803 | 
# |      |6.25e-2    |  1.1149e-2   |  2.0078  |  3.6778e-2  | 1.9608 | 4.7927e-2 | 1.9719 | 
# |  3   |5.00e-1    |  6.7383e-1   |    ----  |  1.2455e-0  |  ----  | 1.9194e-0 |  ----  | 
# |      |2.50e-1    |  1.8124e-1   |  1.8945  |  5.1247e-1  | 1.2812 | 6.9370e-1 | 1.4683 | 
# |      |1.25e-1    |  4.4849e-2   |  2.0148  |  1.4327e-1  | 1.8387 | 1.8812e-1 | 1.8827 | 
# |      |6.25e-2    |  1.1150e-2   |  2.0080  |  3.6784e-2  | 1.9616 | 4.7934e-2 | 1.9725 |
# |  4   |5.00e-1    |  6.7355e-1   |    ----  |  1.2438e-0  |  ----  | 1.9173e-0 |  ----  | 
# |      |2.50e-1    |  1.8123e-1   |  1.8940  |  5.1238e-1  | 1.2795 | 6.9361e-1 | 1.4669 |      
# |      |1.25e-1    |  4.4849e-2   |  2.0148  |  1.4326e-1  | 1.8386 | 1.8811e-1 | 1.8825 | 
# |      |6.25e-2    |  1.1150e-2   |  2.0080  |  3.6784e-2  | 1.9615 | 4.7934e-2 | 1.9725 |  

  


# Generalized alpha method with alpha_f=0.4, alpha_m=0.2 ,
# T=1, xi=1, h=k, CG p in space


# |  p   |     k     | L2 norm (u)   |  rate   | L2 norm (v) |  rate  |  L2 norm  |  rate  |
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|:---------:|:------:|
# |  2   |5.00e-1    |  4.7233e-1   |   ----   |  1.2283e-0  |  ----  | 1.7006e-0 |  ----  | 
# |      |2.50e-1    |  1.2298e-1   |  1.9414  |  5.4672e-1  | 1.1678 | 6.6970e-1 | 1.3445 |                                  
# |      |1.25e-1    |  2.8303e-2   |  2.1194  |  1.5308e-1  | 1.8365 | 1.8138e-1 | 1,8845 | 
# |      |6.25e-2    |  6.8191e-3   |  2.0533  |  3.9282e-2  | 1.9623 | 4.6101e-2 | 1.9761 | 
# |  3   |5.00e-1    |  4.7592e-1   |    ----  |  1.2411e-0  |  ----  | 1.7170e-0 |  ----  | 
# |      |2.50e-1    |  1.2318e-1   |  1.9500  |  5.4793e-1  | 1.1796 | 6.7111e-1 | 1.3553 | 
# |      |1.25e-1    |  2.8314e-2   |  2.1212  |  1.5316e-1  | 1.8390 | 1.8148e-1 | 1.8867 | 
# |      |6.25e-2    |  6.8198e-3   |  2.0537  |  3.9288e-2  | 1.9629 | 4.6108e-2 | 1.9767 |
# |  4   |5.00e-1    |  4.7574e-1   |    ----  |  1.2397e-0  |  ----  | 1.7155e-0 |  ----  | 
# |      |2.50e-1    |  1.2317e-1   |  1.9495  |  5.4785e-1  | 1.1781 | 6.7103e-1 | 1.3542 |      
# |      |1.25e-1    |  2.8314e-2   |  2.1211  |  1.5316e-1  | 1.8387 | 1.8148e-1 | 1.8866 | 
# |      |6.25e-2    |  6.8198e-3   |  2.0537  |  3.9288e-2  | 1.9629 | 4.6108e-2 | 1.9767 |  




# Computational times for the time loop 
# |  p   |     k     |  1 time step |  total   | L2 norm (v) |  rate  | 
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|
# |  2   |5.00e-1    |  3.62e-2s    | 2.60e-1s |  1.2283e-0  |  ----  | 
# |      |2.50e-1    |  2.78e-2s    | 2.96e-1s |  5.4672e-1  | 1.1678 |                        
# |      |1.25e-1    |  2.26e-2s    | 3.67e-1s |  1.5308e-1  | 1.8365 | 
# |      |6.25e-2    |  2.18e-2s    | 5.20e-1s |  3.9282e-2  | 1.9623 | 


