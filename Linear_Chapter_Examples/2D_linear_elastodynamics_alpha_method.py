from fenics import *
import numpy as np 
import matplotlib.pyplot as plt
import timeit


# We consider the 2D linear elastodynamics pproblem
# rho *u_tt+2*rho*xi*u_t+rho*xi^2*u-div (sigma)=  f in  Omega X(0,T],
# u = 0                                             on  \partial\Omega X(0,T],
# u_0=[0, 0]^T , u_1=sqrt{2}*pi*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T.
# Here sigma = 2*mu*epsilon + lambda *tr(epsilon)*Id, with epsilon(u)=1/2*(grad u+ (grad u)^T).
# We set the parameters rho=1, lambda=mu=1, xi=1. 
# The exact solution is given by 
# u=sin(sqrt{2}*pi*t)*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T.
# and f=[(2*pi^2+xi^2)*sin(sqrt{2}*pi*t)+2*sqrt{2}*xi*pi*cos(sqrt{2}*pi*t)]*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T+ 2pi^2*sin(sqrt{2}*pi*t)*[sin(2*pi*y)*cos(2*pi*x), -sin(2*pi*x)*cos(2*pi*y)]^T.
 



#-----------------------------------------------------------------------------
# Spatial discretization and time-stepping  parameters

nx    = 8             # Number of subintervals in the spatial domain  
tmax    = 1           # Max time 
xmax  = 1             # Max point of x domain 
h= xmax/nx            # calculate the uniform mesh-size 
dt= h**2              # time step size 
Nsteps=int(tmax/dt)   # initialize time axis
xi=Constant(1.0)      # Set the parameter xi=1
print('the spatial discretization parameter is', h, 'the temporal discretization parameter is', dt)

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant (0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Elastic parameters
rho= Constant(1.0)
lmbda= Constant(1.0)
mu= Constant(1.0)

# Create mesh and define function space
mesh = UnitSquareMesh(nx,nx)

# Define function space for displacement, velocity and acceleration
V = VectorFunctionSpace(mesh, "CG", 1)

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")

# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)


#  Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant((0.0,0.0)), boundary)

# Exact solutions and RHS 
u_exact= Expression(("-sin(sqrt(2)*pi*t)*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "sin(sqrt(2)*pi*t)*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])"), degree=6, t=0)
v_exact= Expression(("-sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])"), degree=6,t=0)
# f = Expression(("(2*pow(pi,2)*sin(sqrt(2)*pi*t)+2*sqrt(2)*pi*cos(sqrt(2)*pi*t))*(-pow(sin(pi*x[0]),2)*sin(2*pi*x[1]))+sin(sqrt(2)*pi*t)*(-pow(sin(pi*x[0]),2)*sin(2*pi*x[1]))+2*pi**2*sin(sqrt(2)*pi*t)*sin(2*pi*x[1])*cos(2*pi*x[0])", "(2*pi**2*sin(sqrt(2)*pi*t)+2*sqrt(2)*pi*cos(sqrt(2)*pi*t))*sin(2*pi*x[0])*pow(sin(pi*x[1]),2)+sin(sqrt(2)*pi*t)*sin(2*pi*x[0])*pow(sin(pi*x[1]),2)+2*pi**2*sin(sqrt(2)*pi*t)*(-sin(2*pi*x[0])*cos(2*pi*x[1]))"), degree=6, t=0)
f =Expression(("(2*pow(pi,2)*sin(sqrt(2)*pi*t)+2*sqrt(2)*pi*cos(sqrt(2)*pi*t)+sin(sqrt(2)*pi*t))*(-pow(sin(pi*x[0]),2)*sin(2*pi*x[1]))+2*pow(pi,2)*sin(sqrt(2)*pi*t)*sin(2*pi*x[1])*cos(2*pi*x[0])", "(2*pow(pi,2)*sin(sqrt(2)*pi*t)+2*sqrt(2)*pi*cos(sqrt(2)*pi*t)+sin(sqrt(2)*pi*t))*sin(2*pi*x[0])*pow(sin(pi*x[1]),2)+2*pow(pi,2)*sin(sqrt(2)*pi*t)*(-sin(2*pi*x[0])*cos(2*pi*x[1]))"), degree=6, t=0)


# Mass form
def m(u, u_):
    return rho*inner(u, u_)*dx

# Stress tensor 
def sigma(r):
    return 2*mu*sym(nabla_grad(r))+lmbda*tr(sym(nabla_grad(r)))*Identity(len(r))

# Elastic stiffness form
def k(u, u_):
    return inner(sigma(u), sym(grad(u_)))*dx

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
# |  2   |5.00e-1    |  2.6094e-1   |   ----   |  1.5048e-0  |  ----  | 1.7657e-0 |  ----  | 
# |      |2.50e-1    |  3.3472e-2   |  2.9627  |  6.3650e-1  | 1.2413 | 6.6997e-1 | 1.3981 |                                  
# |      |1.25e-1    |  1.3037e-2   |  1.3603  |  1.4093e-1  | 2.1752 | 1.5397e-1 | 2.1214 | 
# |      |6.25e-2    |  3.9799e-3   |  1.7118  |  3.1904e-2  | 2.1432 | 3.5884e-2 | 2.1012 | 
# |  3   |5.00e-1    |  2.2256e-1   |    ----  |  2.1261e-0  |  ----  | 2.3486e-0 |  ----  | 
# |      |2.50e-1    |  1.9691e-2   |  3.4986  |  7.8693e-1  | 1.4339 | 8.0663e-1 | 1.5418 | 
# |      |1.25e-1    |  1.4668e-2   |  0.4249  |  1.5211e-1  | 2.3711 | 1.6678e-1 | 2.2740 | 
# |      |6.25e-2    |  4.1213e-3   |  1.8315  |  3.2574e-2  | 2.2233 | 3.6695e-2 | 2.1843 |
# |  4   |5.00e-1    |  2.2192e-1   |    ----  |  2.1747e-0  |  ----  | 2.3966e-0 |  ----  | 
# |      |2.50e-1    |  1.9713e-2   |  3.4928  |  7.9601e-1  | 1.4500 | 8.1573e-1 | 1.5548 |      
# |      |1.25e-1    |  1.4692e-2   |  0.4241  |  1.5234e-1  | 2.3855 | 1.6703e-1 | 2.2880 | 
# |      |6.25e-2    |  4.1224e-3   |  1.8335  |  3.2581e-2  | 2.2252 | 3.6703e-2 | 2.1861 |  

  


# Generalized alpha method with alpha_f=0.4, alpha_m=0.2 ,
# T=1, xi=1, h=k, CG p in space


# |  p   |     k     | L2 norm (u)   |  rate   | L2 norm (v) |  rate  |  L2 norm  |  rate  |
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|:---------:|:------:|
# |  2   |5.00e-1    |  2.3228e-1   |   ----   |  3.0788e-0  |  ----  | 3.3111e-0 |  ----  | 
# |      |2.50e-1    |  7.0692e-2   |  1.7162  |  1.2725e-0  | 1.2747 | 1.3432e-0 | 1.3016 |                                  
# |      |1.25e-1    |  3.9066e-2   |  0.8556  |  2.5702e-1  | 2.3077 | 2.9609e-1 | 2.1816 | 
# |      |6.25e-2    |  1.1166e-2   |  1.8068  |  5.3009e-2  | 2.2776 | 6.4175e-2 | 2.2060 | 
# |  3   |5.00e-1    |  1.9775e-1   |    ----  |  4.0431e-0  |  ----  | 4.2408e-0 |  ----  | 
# |      |2.50e-1    |  7.9049e-2   |  1.3229  |  1.4423e-0  | 1.4871 | 1.5214e-0 | 1.4789 | 
# |      |1.25e-1    |  4.1102e-2   |  0.9435  |  2.6924e-1  | 2.4214 | 3.1034e-1 | 2.2934 | 
# |      |6.25e-2    |  1.1325e-2   |  1.8597  |  5.3705e-2  | 2.3258 | 6.5030e-2 | 2.2547 |
# |  4   |5.00e-1    |  1.9907e-1   |    ----  |  4.1123e-0  |  ----  | 4.3113e-0 |  ----  | 
# |      |2.50e-1    |  7.9448e-2   |  1.3252  |  1.4516e-0  | 1.5023 | 1.5310e-0 | 1.4936 |      
# |      |1.25e-1    |  4.1126e-2   |  0.9500  |  2.6946e-1  | 2.4295 | 3.1059e-1 | 2.3014 | 
# |      |6.25e-2    |  1.1326e-2   |  1.8604  |  5.3712e-2  | 2.3268 | 6.5038e-2 | 2.2557 |  


# Computational times for the time loop 
# |  p   |     k     |  1 time step |  total   | L2 norm (v) |  rate  | 
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|
# |  2   |5.00e-1    |  6.60e-2s    | 3.89e-1s |  3.3111e-0  |  ----  | 
# |      |2.50e-1    |  1.27e-1s    | 7.74e-1s |  1.3432e-0  | 1.3016 |                        
# |      |1.25e-1    |  3.86e-1s    | 3.44e-0s |  2.9609e-1  | 2.1816 | 
# |      |6.25e-2    |  1.41e-0s    | 2.32e+1s |  6.4175e-2  | 2.2060 | 
