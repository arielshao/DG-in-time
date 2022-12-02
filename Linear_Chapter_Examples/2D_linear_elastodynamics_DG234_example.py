# Import all necessary libraries 
from fenics import *
import numpy as np
from matplotlib.pyplot import*
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
from scipy.integrate import quad
import timeit


# We consider the 2D linear elastodynamics problem
# rho *u_tt+2*rho*xi*u_t+rho*xi^2*u-div (sigma)=  f in  Omega X(0,T],
# u = 0                                             on  \partial\Omega X(0,T],
# u_0=[0, 0]^T , u_1=sqrt{2}*pi*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T.
# Here sigma = 2*mu*epsilon + lambda *tr(epsilon)*Id, with epsilon(u)=1/2*(grad u+ (grad u)^T).
# We set the parameters rho=1, lambda=mu=1, xi=1. 
# The exact solution is given by 
# u=sin(sqrt{2}*pi*t)*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T.
# and f=[(2*pi^2+xi^2)*sin(sqrt{2}*pi*t)+2*sqrt{2}*xi*pi*cos(sqrt{2}*pi*t)]*[-sin^2(pi*x)*sin(2*pi*y), sin(2*pi*x)*sin^2(pi*y)]^T+ 2pi^2*sin(sqrt{2}*pi*t)*[sin(2*pi*y)*cos(2*pi*x), -sin(2*pi*x)*cos(2*pi*y)]^T.

tic=timeit.default_timer()
#Initialization of set 
#-----------------------------------------------------------------------------
# Basic parameters
nx    = 8            # Number of subintervals in the spatial domain  
tmax  = 1            # Max time 
xmax  = 1            # Max point of x domain 

h= xmax/nx           # calculate the uniform mesh-size 
k= h                 # time step size 
nt=int(tmax/k+1)     # initialize time axis
time= np.arange(0, nt) * k
print('The spatial mesh size is', h)
print('The time step is',k, 'with', nt, 'time steps')


# Create mesh and define function space
mesh = UnitSquareMesh(nx,nx)
V = VectorFunctionSpace(mesh, "CG", 2)  
u = TrialFunction(V)
v = TestFunction(V)
#  Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant((0.0,0.0)), boundary)

# Elastic parameters
rho= Constant(1.0)
lmbda= Constant(1.0)
mu= Constant(1.0)

# defining the source term
u_D=      Expression(("sin(sqrt(2)*pi*t)*(-pow(sin(pi*x[0]),2)*sin(2*pi*x[1]))", "sin(sqrt(2)*pi*t)*sin(2*pi*x[0])*pow(sin(pi*x[1]),2)"), degree=6,t=0)
u_Dprime= Expression(("sqrt(2)*pi*cos(sqrt(2)*pi*t)*(-pow(sin(pi*x[0]),2)*sin(2*pi*x[1]))", "sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(2*pi*x[0])*pow(sin(pi*x[1]),2)"), degree=6,t=0)

f_space1 =Expression(("-pow(sin(pi*x[0]),2)*sin(2*pi*x[1])", "sin(2*pi*x[0])*pow(sin(pi*x[1]),2)"), degree=6)
f_space2= Expression(("sin(2*pi*x[1])*cos(2*pi*x[0])","-sin(2*pi*x[0])*cos(2*pi*x[1])"), degree=6)

# Stress tensor 
def sigma(r):
    return 2*mu*sym(nabla_grad(r))+lmbda*tr(sym(nabla_grad(r)))*Identity(len(r))

# Mass form
m= rho*inner(u,v)*dx

# Elastic stiffness form
k1=inner(sigma(u), sym(nabla_grad(v)))*dx 

# RHS
f1=dot(v, f_space1)*dx
f2= dot(v, f_space2)*dx

# Assemble the relevant matrices and RHS vectors
M, F1= assemble_system(m, f1, bc)         
K, F2= assemble_system(k1, f2,bc)                                                                           
M_array= M.array()
K_array= K.array()
F1_array= F1.get_local()
F2_array= F2.get_local()

u_space= Expression(("-sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])"), degree=6)
u_e = interpolate(u_space,V).vector().get_local()

xi=1  # Set the parameter xi=1

D= xi**2*M_array
d=np.size(F1_array)
print('the size d is', d)

# Construct the relevant matrices for the resulting ODE systems
Z0=np.zeros(d);
Z1=np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2).real @u_e
L= 2*xi*np.identity(d)
K= fractional_matrix_power(M_array,-1/2).real@ ((D+K_array) @fractional_matrix_power(M_array,-1/2).real)

# Exact solutions 
Z_e = lambda  t: fractional_matrix_power(M_array,1/2).real@u_e*np.sin(np.sqrt(2)*np.pi*t)
Zprime_e = lambda  t: np.sqrt(2)*np.pi*np.cos(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_array,1/2).real@u_e


# Use 2nd order finite-element in time (with shifted Legendre polynomials as basis functions)
Dt=2+1;
#Assemble the matrices
M1=np.zeros((Dt,Dt), dtype=float);
M1[1,2]=24/(k**2);

M2=np.zeros((Dt,Dt),dtype=float); 
M2[Dt-1,Dt-1]=12/k; M2[Dt-2,Dt-2]=4/k;

M3=np.zeros((Dt,Dt),dtype=float); 
M3[1,0]=2;M3[2,1]=2;

M4=np.zeros((Dt,Dt), dtype=float); 
M4[Dt-1,Dt-1]=36/(k**2); M4[Dt-2,Dt-2]=4/(k**2);
M4[1,2]=-12/(k**2); M4[2,1]=-12/(k**2);

M5=np.zeros((Dt,Dt),dtype=float); 
M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1;
M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1;
M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1;
M=M1+M4

#Construct the matrix A=A_1+B
#Compute the block diagonal matrix A1
A_1=M   
for i in range(1,d):
    A_1= block_diag(A_1,M)    
#Compute the block matrix B
B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])
A=A_1+B; 

#Initialisation
Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
Zprime_minus= np.zeros([nt,d])     #intialize  dZ/dt(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
Zprime_plus = np.zeros([nt,d])     #intialize  dZ/dt(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
Z_minus[0,:]=Z0
Zprime_minus[0,:]=Z1
b=np.zeros(Dt*d)
print(np.size(b))


# Iterations DG2
error_energy=0;
Gsin=(2*(np.pi**2)+xi**2)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)+2*np.pi**2*(fractional_matrix_power(M_array,-1/2).real@ F2_array)
Gcos=(2*np.sqrt(2)*xi*np.pi)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)
sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;

for it in range(1, nt):
    print(it)
    t=time[it]
    u_D.t=t 
    u_Dprime.t=t 
    def phi_t(t,i):
        if i==0:
            return 0;
        if i==1:
            return 2/k;
        if i==2:
            return 12*(t-time[it-1])/(k**2)-6/k;
    sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    b[0::Dt]= (K@Z_minus[it-1,:])
    b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
    b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
    alpha=np.linalg.solve(A,b)
    integrand=lambda t: 2*xi*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)-np.sqrt(2)*np.pi*np.cos(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_array,1/2).real@u_e, alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)-np.sqrt(2)*np.pi*np.cos(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_array,1/2).real@u_e)
    Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]
    Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]
    Zprime_minus[it,:]=  alpha[1::Dt]*2/k+alpha[2::Dt]*6/k
    Zprime_plus[it-1,:]= alpha[1::Dt]*2/k-alpha[2::Dt]*6/k
    error_energy= 1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
    print('the error in energy norm is', np.sqrt(error_energy))
    # Compute the error in L2 norm
    U_end_DG= fractional_matrix_power(M_array,-1/2).real@ Z_minus[it,:]
    Uprime_end_DG= fractional_matrix_power(M_array,-1/2).real@ Zprime_minus[it,:]

    u_end_DG= Function(V)
    u_end_DG.vector().set_local(U_end_DG)

    uprime_end_DG= Function(V)
    uprime_end_DG.vector().set_local(Uprime_end_DG)

    error1=errornorm(u_D, u_end_DG, 'L2')
    error2=errornorm(u_Dprime, uprime_end_DG, 'L2')
    Error=error1+error2
    print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', Error )
toc=timeit.default_timer()
print(toc-tic, 'sec Elapsed')




# # Use 3rd order finite-element in time (with shifted Legendre polynomials as basis functions)
# Dt=3+1;
# # Assemble the matrices
# M1=np.zeros((Dt,Dt), dtype=float); 
# M1[1,2]=24/(k**2); M1[2,3]=120/(k**2); M1[3,2]=24/(k**2);

# M2=np.zeros((Dt,Dt),dtype=float);
# M2[1,1]=4/k; M2[1,3]=4/k; M2[2,2]=12/k; M2[3,1]=4/k; M2[3,3]=24/k;

# M3=np.zeros((Dt,Dt),dtype=float);
# M3[1,0]=2;M3[2,1]=2;M3[3,2]=2; M3[3,0]=2

# M4=np.zeros((Dt,Dt), dtype=float);
# M4[1,1]=4/(k**2); M4[1,2]=-12/(k**2); M4[1,3]=24/(k**2);
# M4[2,1]=-12/(k**2); M4[2,2]=36/(k**2); M4[2,3]=-72/(k**2);
# M4[3,1]=24/(k**2); M4[3,2]=-72/(k**2); M4[3,3]=144/(k**2);

# M5=np.zeros((Dt,Dt),dtype=float); 
# M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1; M5[0,3]=-1;
# M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1; M5[1,3]=1;
# M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1; M5[2,3]=-1;
# M5[3,0]=-1; M5[3,1]=1; M5[3,2]=-1; M5[3,3]=1;

# M=M1+M4

# #Construct the matrix A=A_1+B

# #Compute the block diagonal matrix A1
# A_1=M   
# for i in range(1,d):
#     A_1= block_diag(A_1,M)    
# #Compute the block matrix B
# B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])


# A=A_1+B; 
# # Initialisation
# Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Zprime_minus= np.zeros([nt,d])     #intialize  dZ/dt(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Zprime_plus = np.zeros([nt,d])     #intialize  dZ/dt(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)

# print(np.size(b))
# print(d)

# #Iterations DG3 
# error_energy=0;
# t= 0
# Gsin=(2*(np.pi**2)+xi**2)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)+2*np.pi**2*(fractional_matrix_power(M_array,-1/2).real@ F2_array)
# Gcos=(2*np.sqrt(2)*xi*np.pi)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;
# for it in range(1,nt):
#     print(it)
#     def phi_t(t,i):
#         if i==0:
#             return 0;
#         if i==1:
#             return 2/k;
#         if i==2:
#             return 12*(t-time[it-1])/(k**2)-6/k;
#         if i==3:
#             return 60*(t-time[it-1])**2/(k**3)-60*(t-time[it-1])/(k**2)+12/k;
#     sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     sint3= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     cost3= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     b[0::Dt]= (K@Z_minus[it-1,:])
#     b[1::Dt]= -(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
#     b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
#     b[3::Dt]= -(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*12/k+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos
#     alpha=np.linalg.solve(A,b)
#     integrand= lambda t: 2*xi*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)-np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2).real@u_e*np.cos(np.sqrt(2)*np.pi*t),(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)-np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2).real@u_e*np.cos(np.sqrt(2)*np.pi*t)))
#     Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]
#     Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]
#     Zprime_minus[it,:]=  alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k
#     Zprime_plus[it-1,:]= alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))



# # Use 4th order finite-element in time (with shifted Legendre polynomials as basis functions)
# Dt=4+1;
# # Assemble the matrices
# M1=np.zeros((Dt,Dt), dtype=float); 
# M1[1,2]=24/(k**2); M1[1,4]=80/(k**2);
# M1[2,3]=120/(k**2); 
# M1[3,2]=24/(k**2);M1[3,4]=360/(k**2);
# M1[4,3]=120/(k**2);

# M2=np.zeros((Dt,Dt),dtype=float);
# M2[1,1]=4/k; M2[1,3]=4/k;
# M2[2,2]=12/k; M2[2,4]=12/k;
# M2[3,1]=4/k; M2[3,3]=24/k;
# M2[4,2]=12/k; M2[4,4]=40/k;

# M3=np.zeros((Dt,Dt),dtype=float);
# M3[1,0]=2;M3[2,1]=2;M3[3,2]=2; M3[3,0]=2
# M3[4,1]=2; M3[4,3]=2;

# M4=np.zeros((Dt,Dt), dtype=float);
# M4[1,1]=  4/(k**2); M4[1,2]=-12/(k**2); M4[1,3]=24/(k**2);  M4[1,4]=-40/(k**2);
# M4[2,1]=-12/(k**2); M4[2,2]=36/(k**2);  M4[2,3]=-72/(k**2); M4[2,4]=120/(k**2);
# M4[3,1]= 24/(k**2); M4[3,2]=-72/(k**2); M4[3,3]=144/(k**2); M4[3,4]=-240/(k**2);
# M4[4,1]=-40/(k**2); M4[4,2]=120/(k**2); M4[4,3]=-240/(k**2);M4[4,4]=400/(k**2);


# M5=np.zeros((Dt,Dt),dtype=float); 
# M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1; M5[0,3]=-1; M5[0,4]=1;
# M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1; M5[1,3]=1; M5[1,4]=-1;
# M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1; M5[2,3]=-1; M5[2,4]=1;
# M5[3,0]=-1; M5[3,1]=1; M5[3,2]=-1; M5[3,3]=1; M5[3,4]=-1;
# M5[4,0]= 1; M5[4,1]=-1; M5[4,2]=1; M5[4,3]=-1; M5[4,4]=1;

# M=M1+M4

# #Construct the matrix A=A_1+B

# #Compute the block diagonal matrix A1
# A_1=M   
# for i in range(1,d):
#     A_1= block_diag(A_1,M)    
# #Compute the block matrix B
# B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])
# A=A_1+B; 


# #Initialisation
# Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Zprime_minus= np.zeros([nt,d])     #intialize  dZ/dt(t_n^{-}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Zprime_plus = np.zeros([nt,d])     #intialize  dZ/dt(t_n^{+}, x_m) for n=0,1,...,N, m=1/2,1,...,M-1, M-1/2
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)


# # Iterations DG4
# t=0
# error_energy=0;
# Gsin=(2*(np.pi**2)+xi**2)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)+2*np.pi**2*(fractional_matrix_power(M_array,-1/2).real@ F2_array)
# Gcos=(2*np.sqrt(2)*xi*np.pi)*(fractional_matrix_power(M_array,-1/2).real@ F1_array)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;
# for it in range(1,nt):
#     print(it)
    
#     def phi_t(t,i):
#         if i==0:
#             return 0;
#         if i==1:
#             return 2/k;
#         if i==2:
#             return 12*(t-time[it-1])/(k**2)-6/k;
#         if i==3:
#             return 60*(t-time[it-1])**2/(k**3)-60*(t-time[it-1])/(k**2)+12/k;
#         if i==4:
#             return 280*(t-time[it-1])**3/k**4-420*(t-time[it-1])**2/k**3+180*(t-time[it-1])/k**2-20/k;   
#     sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     sint3= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     cost3= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     sint4= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,4);
#     cost4= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,4);
#     b[0::Dt]= (K@Z_minus[it-1,:])
#     b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
#     b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
#     b[3::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*12/k+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos
#     b[4::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-20/k)+quad(sint4, time[it-1], time[it])[0]*Gsin+quad(cost4, time[it-1], time[it])[0]*Gcos
#     alpha=np.linalg.solve(A,b)
#     integrand= lambda t: 2*xi*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2).real@u_e*np.cos(np.sqrt(2)*np.pi*t), alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2).real@u_e*np.cos(np.sqrt(2)*np.pi*t))
#     Z_minus[it,:]= alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]+alpha[4::Dt]
#     Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]+alpha[4::Dt]
#     Zprime_minus[it,:]=  alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k+alpha[4::Dt]*(20/k)
#     Zprime_plus[it-1,:]= alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k-alpha[4::Dt]*(20/k)
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))



# # Use 5th order finite-element in time (with shifted Legendre polynomials as basis functions)
# Dt=5+1;
# # Assemble the matrices
# M1=np.zeros((Dt,Dt), dtype=float); 
# M1[1,2]=24/(k**2);M1[1,4]=80/(k**2);
# M1[2,3]=120/(k**2); M1[2,5]=336/(k**2)
# M1[3,2]=24/(k**2);M1[3,4]=360/(k**2); 
# M1[4,3]=120/(k**2);M1[4,5]=840/(k**2);
# M1[5,2]=24/(k**2); M1[5,4]=360/(k**2);

# M2=np.zeros((Dt,Dt),dtype=float);
# M2[1,1]=4/k; M2[1,3]=4/k;M2[1,5]=4/k;
# M2[2,2]=12/k; M2[2,4]=12/k;
# M2[3,1]=4/k; M2[3,3]=24/k;M2[3,5]=24/k;
# M2[4,2]=12/k; M2[4,4]=40/k;
# M2[5,1]=4/k;M2[5,3]=24/k;M2[5,5]=60/k;

# M3=np.zeros((Dt,Dt),dtype=float);
# M3[1,0]=2;M3[2,1]=2;M3[3,2]=2; M3[3,0]=2
# M3[4,1]=2;M3[4,3]=2;  
# M3[5,0]=2;M3[5,2]=2;M3[5,4]=2;

# M4=np.zeros((Dt,Dt), dtype=float);
# M4[1,1]=4/(k**2);   M4[1,2]=-12/(k**2); M4[1,3]=24/(k**2);M4[1,4]=-40/(k**2);M4[1,5]=60/(k**2);
# M4[2,1]=-12/(k**2); M4[2,2]=36/(k**2); M4[2,3]=-72/(k**2);M4[2,4]=120/(k**2);M4[2,5]=-180/(k**2);
# M4[3,1]=24/(k**2);  M4[3,2]=-72/(k**2); M4[3,3]=144/(k**2);M4[3,4]=-240/(k**2);M4[3,5]=360/(k**2);
# M4[4,1]=-40/(k**2); M4[4,2]=120/(k**2); M4[4,3]=-240/(k**2);M4[4,4]=400/(k**2);M4[4,5]=-600/(k**2);
# M4[5,1]=60/(k**2);  M4[5,2]=-180/(k**2); M4[5,3]=360/(k**2);M4[5,4]=-600/(k**2); M4[5,5]=900/(k**2);


# M5=np.zeros((Dt,Dt),dtype=float); 
# M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1; M5[0,3]=-1; M5[0,4]=1; M5[0,5]=-1;
# M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1; M5[1,3]=1; M5[1,4]=-1;M5[1,5]=1;
# M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1; M5[2,3]=-1; M5[2,4]=1; M5[2,5]=-1;
# M5[3,0]=-1; M5[3,1]=1; M5[3,2]=-1; M5[3,3]=1; M5[3,4]=-1;M5[3,5]=1;
# M5[4,0]= 1; M5[4,1]=-1;M5[4,2]=1; M5[4,3]=-1; M5[4,4]=1; M5[4,5]=-1;
# M5[5,0]=-1; M5[5,1]=1; M5[5,2]=-1; M5[5,3]=1; M5[5,4]=-1; M5[5,5]=1;

# M=M1+M4

# #Construct the matrix A=A_1+B

# #Compute the block diagonal matrix A1
# A_1=M   
# for i in range(1,d):
#     A_1= block_diag(A_1,M)    
# #Compute the block matrix B
# B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])


# A=A_1+B; 

# #Initialisation
# Z_minus=np.zeros([nt,d])           
# Z_plus= np.zeros([nt,d])          
# Zprime_minus= np.zeros([nt,d])     
# Zprime_plus = np.zeros([nt,d])     
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)

# # Iterations DG5
# error_energy=0;
# Gsin=(2*(np.pi**2)+xi**2)*(fractional_matrix_power(M_bar,-1/2).real@ F1_space)+2*np.pi**2*(fractional_matrix_power(M_bar,-1/2).real@ F2_space)
# Gcos=(2*np.sqrt(2)*xi*np.pi)*(fractional_matrix_power(M_bar,-1/2).real@ F1_space)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;
# for it in range(1,nt):
#     print(it)
#     def phi_t(t,i):
#         if i==0:
#             return 0;
#         if i==1:
#             return 2/k;
#         if i==2:
#             return 12*(t-time[it-1])/(k**2)-6/k;
#         if i==3:
#             return 60*(t-time[it-1])**2/(k**3)-60*(t-time[it-1])/(k**2)+12/k;
#         if i==4:
#             return 280*(t-time[it-1])**3/k**4-420*(t-time[it-1])**2/k**3+180*(t-time[it-1])/k**2-20/k; 
#         if i==5:
#             return 1260*(t-time[it-1])**4/k**5-2520*(t-time[it-1])**3/k**4+1680*(t-time[it-1])**2/k**3-420*(t-time[it-1])/k**2+30/k
#     sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     sint3= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     cost3= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,3);
#     sint4= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,4);
#     cost4= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,4);
#     sint5= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,5);
#     cost5= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,5);
#     b[0::Dt]= (K@Z_minus[it-1,:])
#     b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
#     b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
#     b[3::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*12/k+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos
#     b[4::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-20/k)+quad(sint4, time[it-1], time[it])[0]*Gsin+quad(cost4, time[it-1], time[it])[0]*Gcos
#     b[5::Dt]= -(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(30/k)+quad(sint5, time[it-1], time[it])[0]*Gsin+quad(cost5, time[it-1], time[it])[0]*Gcos
#     alpha=np.linalg.solve(A,b)
#     integrand= lambda t: 2*xi*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)+alpha[5::Dt]*phi_t(t,5)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar,1/2).real@u_space*np.cos(np.sqrt(2)*np.pi*t), alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)+alpha[5::Dt]*phi_t(t,5)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar,1/2).real@u_space*np.cos(np.sqrt(2)*np.pi*t))
#     Z_minus[it,:]= alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]+alpha[4::Dt]+alpha[5::Dt]
#     Z_plus[it-1,:]=alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]+alpha[4::Dt]-alpha[5::Dt]
#     Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k+alpha[4::Dt]*(20/k)+alpha[5::Dt]*30/k
#     Zprime_plus[it-1,:]=alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k-alpha[4::Dt]*(20/k)+alpha[5::Dt]*30/k
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))



# T=1, xi=1, h=k, CG q in space


# |  q   |     k     | energy norm |  rate  |   L2 norm  |  rate  |
# |------|:---------:|:-----------:|:------:|:----------:|:------:|
# |  2   |5.00e-1    |  1.2668e-0  |  ----  | 7.2172e-1  |  ----  |
# |      |2.50e-1    |  5.2863e-1  | 1.2609 | 9.5802e-2  | 2.9133 |                                      
# |      |1.25e-2    |  1.9754e-1  | 1.4201 | 1.2390e-2  | 2.9509 |
# |      |1.00e-2    |  1.4262e-1  | 1.4599 | 6.8663e-3  | 2.6452 |
# |  3   |5.00e-1    |  3.1328e-1  |  ----  | 1.3788e-1  |  ----  | 
# |      |2.50e-1    |  6.0999e-2  | 2.3606 | 1.2789e-2  | 3.4304 |
# |      |1.25e-2    |  1.0548e-2  | 2.5318 | 6.1569e-4  | 4.3765 |
# |      |1.00e-2    |  6.0522e-3  | 2.4895 | 2.4334e-4  | 4.1600 |
# |  4   |5.00e-1    |  1.5241e-1  |  ----  | 8.4535e-2  |  ----  |
# |      |2.50e-1    |  6.1539e-3  | 4.6303 | 1.7324e-3  | 5.6087 |        
# |      |1.25e-1    |  4.5846e-4  | 3.7466 | 5.4731e-5  | 4.9843 |    
# |      |1.00e-1    |  2.0732e-4  | 3.5565 | 1.7987e-5  | 4.9868 |   


# # Computed errors |||\mathbf{Z}-\mathbf{Z}_DG||| versus 1/k (loglog scale) and q=2,3,4
# Delta_t=np.array([ 1/5.000e-1, 1/2.500e-1, 1/1.250e-1, 1/1.000e-1 ]);
# error_2= np.array([ 1.2668e-0, 5.2863e-1   , 1.9754e-1, 1.4262e-1 ]);
# error_3= np.array([ 3.1328e-1   , 6.0999e-2,  1.0548e-2,  6.0522e-3 ]);
# error_4= np.array([ 1.5241e-1 , 6.1539e-3 , 4.5846e-4,  2.0732e-4   ]);
# loglog(Delta_t, error_2, label='DG 2', linestyle='-', marker='o', color='blue', linewidth=3)
# loglog(Delta_t, error_3, label='DG 3', linestyle=':', marker='p', color='red', linewidth=3)
# loglog(Delta_t, error_4, label='DG 4', linestyle='--',marker='D', color='black', linewidth=3)
# # loglog(Delta_t, error_5, label='DG 5', linestyle='-.', marker='s',color='green', linewidth=3)
# title('Plot of $|||\mathbf{Z}-\mathbf{Z}_{\mathrm{DG}}|||$ versus 1/k')
# legend(loc = 'best')
# grid()
# savefig('2D_linear_elastodynamics_energy_error.png', dpi=300, bbox_inches='tight')
  
# # Computed errors ||dot u- dot u_DG||+||u-u_DG|| versus 1/k (loglog scale) and q=2,3,4
# Delta_t=np.array([ 1/5.000e-1, 1/2.500e-1, 1/1.250e-1, 1/1.000e-1 ]);
# error_2= np.array([ 7.2172e-1, 9.5802e-2  , 1.2390e-2 , 6.8663e-3  ]);
# error_3= np.array([ 1.3788e-1 , 1.2789e-2 ,  6.1569e-4,  2.4334e-4 ]);
# error_4= np.array([ 8.4535e-2, 1.7324e-3 , 5.4731e-5,  1.7987e-5 ]);
# # error_5= np.array([ 1.2959e-9  , 2.0069e-11 , 5.8305e-12 ,  2.8752e-12]);
# loglog(Delta_t, error_2, label='DG 2', linestyle='-', marker='o', color='blue', linewidth=3)
# loglog(Delta_t, error_3, label='DG 3', linestyle=':', marker='p', color='red', linewidth=3)
# loglog(Delta_t, error_4, label='DG 4', linestyle='--',marker='D', color='black', linewidth=3)
# # loglog(Delta_t, error_5, label='DG 5', linestyle='-.', marker='s',color='green', linewidth=3)
# title('Plot of $||\dot{u}- \dot{u}_\mathrm{DG}||_{L^2(\Omega)}$+$||u- u_\mathrm{DG}||_{L^2(\Omega)}$ versus 1/k')
# legend(loc = 'best')
# grid()
# savefig('2D_linear_elastodynamics_l2_error.png', dpi=300, bbox_inches='tight')




# T=1, xi=1, h=k, CG 2q-2 in space

# |  q   |     k     |  energy norm |  rate  |  L2 norm    |  rate  |
# |------|:---------:| :-----------:|:------:| :----------:|:------:|
# |  2   |5.00e-1    |   1.2668e-0  |  ----  |  7.2172e-1  |  ----  |
# |      |2.50e-1    |   5.2863e-1  | 1.2609 |  9.5802e-2  | 2.9133 |                                      
# |      |1.25e-1    |   1.9754e-1  | 1.4201 |  1.2390e-2  | 2.9509 |
# |      |1.00e-1    |   1.4262e-1  | 1.4599 |  6.8663e-3  | 2.6452 |
# |  3   |5.00e-1    |   2.9135e-1  |  ----  |  7.3446e-2  |  ----  | 
# |      |2.50e-1    |   5.6740e-2  | 2.3603 |  3.7267e-3  | 4.3007 |
# |      |1.25e-1    |   1.0486e-2  | 2.4359 |  1.1836e-4  | 4.9766 |
# |      |1.00e-1    |   6.0356e-3  | 2.4754 |  3.8835e-5  | 4.9941 |
# |  4   |5.00e-1    |   5.2822e-2  |  ----  |  6.7767e-3  |  ----  |
# |      |2.50e-1    |   4.9297e-3  | 3.4216 |  4.4456e-5  | 7.2521 | 
# |      |1.25e-1    |   4.4454e-4  | 3.4711 |  3.7959e-7  | 6.8718 | 



# Computational times for the time loop 
# |  p   |     k     |  1 time step |  total   | L2 norm (v) |  rate  | 
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|
# |  2   |5.00e-1    |  2.51e-1s    | 1.33e-0s |  7.2172e-1  |  ----  | 
# |      |2.50e-1    |  2.94e+1s    | 2.05e+1s |  9.5802e-2  | 2.9133 |                        
# |      |1.25e-1    |  9.81e+2s    | 1.04e+3s |  1.2390e-2  | 2.9509 | 
#
         
