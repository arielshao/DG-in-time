from fenics import *
import numpy as np
import scipy 
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
from scipy.integrate import quad
from matplotlib.pyplot import *
import timeit


# We consider the one-dimensional wave-type equation
# u_tt+2*gamma*u_t+gamma^2*u-u_xx=f in (0,1)x(0,T],
# u(0, t) = u(1,t)=0 for all  t in (0,T], 
# u(x,0)=u_0 and  u_t(x,0)=u_1.

# We set the parameter gamma=1 and let u_0, u_1 and f be chosen such that the exact solution is 
# u=sin(sqrt{2}*pi*t)*sin(pi*x).
# That is, u_0=0, u_1= sqrt{2}*pi*sin(pi x), 
# and f=[(-pi^2+gamma^2)*sin(\sqrt{2}*pi *t)+2*sqrt{2}*gamma*pi *cos(sqrt{2}*pi*t)]*sin(pi* x).


# Initialization of set 
#-----------------------------------------------------------------------------
# Basic parameters
tic=timeit.default_timer()  # Start timer
nx    = 2                   # Number of subintervals in the spatial domain  
tmax    = 1                 # Max time 
xmax  = 1                   # Max point of spatial domain 
h= xmax/nx                  # calculate the uniform mesh-size 
k= h                        # time step size 
nt=int(tmax/k+1)            # initialize time axis
time= np.arange(0, nt)* k
print('h is', h, 'k is', k)

# Create mesh and define function space
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 4)
u = TrialFunction(V)
v = TestFunction(V)

# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0), boundary)

u_D= Expression('sin(pi*x[0])', degree=10)
f_space=u_D

 # The exact displacement u and the exact velocity u_t
u_exact=Expression("sin(sqrt(2)*t*pi)*sin(pi*x[0])", t=0, degree=10)
u_prime= Expression("sqrt(2)*pi*cos(sqrt(2)*t*pi)*sin(pi*x[0])", t=0, degree=10)

# Mass form
m= dot(u,v)*dx
# Stiffness form
k1= dot(grad(u), grad(v))*dx 
# RHS 
f=  dot(v, f_space)*dx

# Assemble the relevant matrices
M_bar, F_space= assemble_system(m, f, bc)                                                                                                                                                
K_bar,_= assemble_system(k1, f,bc)                                                                           

M_bar_array= M_bar.array()
K_bar_array= K_bar.array()
F_space_array= F_space.get_local()
u_space = interpolate(u_D,V).vector().get_local()

gamma=1
C= 2*gamma*M_bar_array
D= gamma**2*M_bar_array

d=np.size(F_space_array)


# Construct the relevant matrices for the resulting ODE systems
Z0=np.zeros(d)
Z1=np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2) @u_space
K= fractional_matrix_power(M_bar_array,-1/2)@ ((D+K_bar_array) @fractional_matrix_power(M_bar_array,-1/2))
L= 2*gamma*np.identity(d)

# Exact solutions 
Z_e = lambda  t: np.sin(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_bar_array,1/2)@u_space
Zprime_e = lambda  t: np.cos(np.sqrt(2)*np.pi*t)*np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space

# # Use 2nd order finite-element in time (with shifted Legendre polynomials as basis functions)
# Dt=2+1;
# #Assemble the matrices
# M1=np.zeros((Dt,Dt),dtype=float);
# M1[1,2]=24/(k**2);
# M2=np.zeros((Dt,Dt),dtype=float); 
# M2[2,2]=12/k; M2[1,1]=4/k;
# M3=np.zeros((Dt,Dt),dtype=float); 
# M3[1,0]=2;M3[2,1]=2;
# M4=np.zeros((Dt,Dt), dtype=float); 
# M4[Dt-1,Dt-1]=36/(k**2); M4[Dt-2,Dt-2]=4/(k**2);
# M4[1,2]=-12/(k**2); M4[2,1]=-12/(k**2);
# M5=np.zeros((Dt,Dt),dtype=float);
# M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1;
# M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1;
# M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1;
# M=M1+M4
# #Compute the block diagonal matrix A1
# A_1=M   
# for i in range(1,d):
#     A_1= block_diag(A_1,M)    
# #Compute the block matrix B
# B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])
# A=A_1+B; 
## Sparsity plot of A
# clf()
# matplotlib.pyplot.spy(A)
# savefig('1D_linear_wave_sparsity_A_DG2.png', dpi=300, bbox_inches='tight')

# #Initialisation
# Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_minus= np.zeros([nt,d])     #intialize  Z'(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_plus = np.zeros([nt,d])     #intialize  Z'(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)
# np.shape(Z_minus)

# # Iterations DG2
# error_energy=0;
# Gsin=(-np.pi**2+gamma**2)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
# Gcos=(2*np.sqrt(2)*gamma*np.pi)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k

# for it in range(1,nt):
#     print(it)
#     t=time[it]
#     u_exact.t=t
#     u_prime.t=t 
#     def phi_t(t,i):
#         if i==0:
#             return 0;
#         if i==1:
#             return 2/k;
#         if i==2:
#             return 12*(t-time[it-1])/(k**2)-6/k;
#     sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
#     b[0::Dt]= (K@Z_minus[it-1,:])
#     b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
#     b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
#     alpha=np.linalg.solve(A,b)
#     integrand=lambda t: 2*gamma*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)-np.sqrt(2)*np.pi*np.cos(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_bar_array,1/2)@u_space, alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)-np.sqrt(2)*np.pi*np.cos(np.sqrt(2)*np.pi*t)*fractional_matrix_power(M_bar_array,1/2)@u_space)
#     Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]
#     Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]
#     Zprime_minus[it,:]=  alpha[1::Dt]*2/k+alpha[2::Dt]*6/k
#     Zprime_plus[it-1,:]= alpha[1::Dt]*2/k-alpha[2::Dt]*6/k
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))
#     # Compute the error in L2 norm
#     U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[it,:]
#     Uprime_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Zprime_minus[it,:]
#     u_end_DG= Function(V)
#     u_end_DG.vector().set_local(U_end_DG)
#     uprime_end_DG= Function(V)
#     uprime_end_DG.vector().set_local(Uprime_end_DG)
#     error1=errornorm(u_exact, u_end_DG, 'L2')
#     error2=errornorm(u_prime, uprime_end_DG, 'L2')
#     error=error1+error2
#     print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
#     print('The L2 error ||dot u- dot u_DG|| is', error1 )
      
# toc=timeit.default_timer()
# print(toc-tic, 'sec Elapsed')  


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
## Sparsity plot of A
# clf()
# matplotlib.pyplot.spy(A)
# savefig('1D_linear_wave_sparsity_A_DG3.png', dpi=300, bbox_inches='tight')


# #Initialisation
# Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_minus= np.zeros([nt,d])     #intialize  dZ/dt(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_plus = np.zeros([nt,d])     #intialize  dZ/dt(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)
# np.shape(Z_minus)
# print(np.size(b))
# print(d)


# #Iterations DG3 
# error_energy=0;
# Gsin=(-np.pi**2+gamma**2)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
# Gcos=(2*np.sqrt(2)*gamma*np.pi)*(fractional_matrix_power(M_bar_array,-1/2)@F_space_array)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k
# for it in range(1,nt):
#     print(it)
#     t=time[it]
#     u_exact.t=t
#     u_prime.t=t 
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
#     b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
#     b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
#     b[3::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*12/k+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos
#     alpha=np.linalg.solve(A,b)
#     integrand= lambda t: 2*gamma*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t),alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t))
#     Z_minus[it,:]=alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]
#     Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]
#     Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k
#     Zprime_plus[it-1,:]=alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))
#     # Compute the error in L2 norm
#     U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[it,:]
#     Uprime_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Zprime_minus[it,:]
#     u_end_DG= Function(V)
#     u_end_DG.vector().set_local(U_end_DG)
#     uprime_end_DG= Function(V)
#     uprime_end_DG.vector().set_local(Uprime_end_DG)
#     error1=errornorm(u_exact, u_end_DG, 'L2')
#     error2=errornorm(u_prime, uprime_end_DG, 'L2')
#     error=error1+error2
#     print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
#     print('The L2 error ||dot u- dot u_DG|| is', error1 )
      
# toc=timeit.default_timer()
# print(toc-tic, 'sec Elapsed')  



# Use 4th order finite-element in time (with shifted Legendre polynomials as basis functions)
Dt=4+1;
# Assemble the matrices
M1=np.zeros((Dt,Dt), dtype=float); 
M1[1,2]=24/(k**2); M1[1,4]=80/(k**2);
M1[2,3]=120/(k**2); 
M1[3,2]=24/(k**2);M1[3,4]=360/(k**2);
M1[4,3]=120/(k**2);

M2=np.zeros((Dt,Dt),dtype=float);
M2[1,1]=4/k; M2[1,3]=4/k;
M2[2,2]=12/k; M2[2,4]=12/k;
M2[3,1]=4/k; M2[3,3]=24/k;
M2[4,2]=12/k; M2[4,4]=40/k;

M3=np.zeros((Dt,Dt),dtype=float);
M3[1,0]=2;M3[2,1]=2;M3[3,2]=2; M3[3,0]=2
M3[4,1]=2; M3[4,3]=2;

M4=np.zeros((Dt,Dt), dtype=float);
M4[1,1]=  4/(k**2); M4[1,2]=-12/(k**2); M4[1,3]=24/(k**2);  M4[1,4]=-40/(k**2);
M4[2,1]=-12/(k**2); M4[2,2]=36/(k**2);  M4[2,3]=-72/(k**2); M4[2,4]=120/(k**2);
M4[3,1]= 24/(k**2); M4[3,2]=-72/(k**2); M4[3,3]=144/(k**2); M4[3,4]=-240/(k**2);
M4[4,1]=-40/(k**2); M4[4,2]=120/(k**2); M4[4,3]=-240/(k**2);M4[4,4]=400/(k**2);


M5=np.zeros((Dt,Dt),dtype=float); 
M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1; M5[0,3]=-1; M5[0,4]=1;
M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1; M5[1,3]=1; M5[1,4]=-1;
M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1; M5[2,3]=-1; M5[2,4]=1;
M5[3,0]=-1; M5[3,1]=1; M5[3,2]=-1; M5[3,3]=1; M5[3,4]=-1;
M5[4,0]= 1; M5[4,1]=-1; M5[4,2]=1; M5[4,3]=-1; M5[4,4]=1;

M=M1+M4

#Construct the matrix A=A_1+B

#Compute the block diagonal matrix A1
A_1=M   
for i in range(1,d):
    A_1= block_diag(A_1,M)    
#Compute the block matrix B
B=np.block([[L[i, j]*M2+K[i,j]*(M3+M5) for j in range(0,d) ] for i in range(0,d)])


A=A_1+B; 
## Sparsity plot of A
# clf()
# matplotlib.pyplot.spy(A)
# savefig('1D_linear_wave_sparsity_A_DG4.png', dpi=300, bbox_inches='tight')
## Check condition number of A
# conNumber=np.linalg.cond(A, p=None)
# print('the condition number of the matrix A is', conNumber)
## Check if A is symmetric
# diff_A= A.transpose()-A
# print(diff_A)

#Initialisation
Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
Zprime_minus= np.zeros([nt,d])     #intialize  Z_t(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
Zprime_plus = np.zeros([nt,d])     #intialize  Z_t(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
Z_minus[0,:]=Z0
Zprime_minus[0,:]=Z1
b=np.zeros(Dt*d)
np.shape(Z_minus)
print(np.size(b))
print(d)


# Iterations DG4
error_energy=0;
error_energy2=0
Gsin=(-np.pi**2+gamma**2)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
Gcos=(2*np.sqrt(2)*gamma*np.pi)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k
for it in range(1,nt):
    print(it)
    t=time[it]
    u_exact.t=t
    u_prime.t=t 
    def phi_t(t,i):
        if i==0:
            return 0;
        if i==1:
            return 2/k;
        if i==2:
            return 12*(t-time[it-1])/(k**2)-6/k;
        if i==3:
            return 60*(t-time[it-1])**2/(k**3)-60*(t-time[it-1])/(k**2)+12/k;
        if i==4:
            return 280*(t-time[it-1])**3/k**4-420*(t-time[it-1])**2/k**3+180*(t-time[it-1])/k**2-20/k; 
    sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    sint3= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,3);
    cost3= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,3);
    sint4= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,4);
    cost4= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,4);
    b[0::Dt]= (K@Z_minus[it-1,:])
    b[1::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos
    b[2::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos
    b[3::Dt]=-(K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*12/k+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos
    b[4::Dt]= (K@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-20/k)+quad(sint4, time[it-1], time[it])[0]*Gsin+quad(cost4, time[it-1], time[it])[0]*Gcos
    alpha=np.linalg.solve(A,b)
    integrand = lambda t: 2*gamma*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t), alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t))
    Z_minus[it,:]= alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]+alpha[4::Dt]
    Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]+alpha[4::Dt]
    Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k+alpha[4::Dt]*(20/k)
    Zprime_plus[it-1,:]=alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k-alpha[4::Dt]*(20/k)
    # U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[nt-1,:]

    error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
   
    print('the error in energy norm is', np.sqrt(error_energy))
    # Compute the error in L2 norm
    U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[it,:]
    Uprime_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Zprime_minus[it,:]
    u_end_DG= Function(V)
    u_end_DG.vector().set_local(U_end_DG)
    uprime_end_DG= Function(V)
    uprime_end_DG.vector().set_local(Uprime_end_DG)
    error1=errornorm(u_exact, u_end_DG, 'L2')
    error2=errornorm(u_prime, uprime_end_DG, 'L2')
    error=error1+error2
    print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
    print('The L2 error ||dot u- dot u_DG|| is', error1 )
      
toc=timeit.default_timer()
print(toc-tic, 'sec Elapsed') 
# Use 5th order finite-element in time (with shifted Legendre polynomials as basis functions)
# Dt=5+1;
# # Assemble the matrices
# M1=np.zeros((Dt,Dt), dtype=float); 
# M1[1,2]=24/(k**2);M1[1,4]=80/(k**2);
# M1[2,3]=120/(k**2); M1[2,5]=336/(k**2)
# M1[3,2]=24/(k**2);M1[3,4]=360/(k**2); 
# M1[4,3]=120/(k**2);M1[4,5]=840/(k**2);
# M1[5,2]=24/(k**2); M1[5,4]=360/(k**2);

# M2=np.zeros((Dt,Dt),dtype=float);
# M2[1,1]=4/k;  M2[1,3]=4/k;M2[1,5]=4/k;
# M2[2,2]=12/k; M2[2,4]=12/k;
# M2[3,1]=4/k;  M2[3,3]=24/k;M2[3,5]=24/k;
# M2[4,2]=12/k; M2[4,4]=40/k;
# M2[5,1]=4/k;  M2[5,3]=24/k;M2[5,5]=60/k;

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
## Sparsity plot of A
# clf()
# matplotlib.pyplot.spy(A)
# savefig('1D_linear_wave_sparsity_A_DG5.png', dpi=300, bbox_inches='tight')

# #Initialisation
# Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_minus= np.zeros([nt,d])     #intialize  dZ/dt(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Zprime_plus = np.zeros([nt,d])     #intialize  dZ/dt(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
# Z_minus[0,:]=Z0
# Zprime_minus[0,:]=Z1
# b=np.zeros(Dt*d)

# # Iterations DG5
# error_energy=0;
# Gsin=(-np.pi**2+gamma**2)*fractional_matrix_power(M_bar_array,-1/2)@ F_space_array
# Gcos=(2*np.sqrt(2)*gamma*np.pi)*(fractional_matrix_power(M_bar_array,-1/2)@ F_space_array)
# sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
# cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;
# for it in range(1,nt):
#     print(it)
#     t=time[it]
#     u_exact.t=t
#     u_prime.t=t 
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
#     integrand= lambda t: 2*gamma*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)+alpha[5::Dt]*phi_t(t,5)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t), alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)+alpha[5::Dt]*phi_t(t,5)-np.sqrt(2)*np.pi*fractional_matrix_power(M_bar_array,1/2)@u_space*np.cos(np.sqrt(2)*np.pi*t))
#     Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]+alpha[4::Dt]+alpha[5::Dt]
#     Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]+alpha[4::Dt]-alpha[5::Dt]
#     Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k+alpha[4::Dt]*(20/k)+alpha[5::Dt]*30/k
#     Zprime_plus[it-1,:]=alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k-alpha[4::Dt]*(20/k)+alpha[5::Dt]*30/k
#     error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
#     print('the error in energy norm is', np.sqrt(error_energy))
#             # Compute the error in L2 norm
#     U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[it,:]
#     Uprime_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Zprime_minus[it,:]
#     u_end_DG= Function(V)
#     u_end_DG.vector().set_local(U_end_DG)
#     uprime_end_DG= Function(V)
#     uprime_end_DG.vector().set_local(Uprime_end_DG)
#     error1=errornorm(u_exact, u_end_DG, 'L2')
#     error2=errornorm(u_prime, uprime_end_DG, 'L2')
#     error=error1+error2
#     print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
#     print('The L2 error ||dot u- dot u_DG|| is', error1 )
      
      
# # Compute the error in energy norm
# energy=error_energy+1/2*(Zprime_minus[nt-1,:]-Zprime_e(time[nt-1]))@(Zprime_minus[nt-1,:]-Zprime_e(time[nt-1]))+1/2*K@(Z_minus[nt-1,:]-Z_e(time[nt-1]))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))
# print('the energy error norm is', np.sqrt(energy))
# # print('the energy error norm without last step is', np.sqrt(error_energy))


# # Compute the error in L2 norm
# U_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Z_minus[nt-1,:]
# Uprime_end_DG= fractional_matrix_power(M_bar_array,-1/2)@ Zprime_minus[nt-1,:]

# u_end_DG= Function(V)
# u_end_DG.vector().set_local(U_end_DG)

# uprime_end_DG= Function(V)
# uprime_end_DG.vector().set_local(Uprime_end_DG)

# # error1=errornorm(u, u_end_DG, 'L2')
# # error2=errornorm(u_prime, uprime_end_DG, 'L2')
# # error=error1+error2
# # print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
# error=errornorm(u_prime, uprime_end_DG, 'L2')
# print('The L2 error ||dot u- dot u_DG|| is', error )



#03/04/2022 Numerical tests

# T=1, gamma=1, h=k, CG q in space

# |  q   |     k     | $L^2$ norm   |  rate    |   energy norm    |  rate   |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|
# |  2   |5.000e-1   |  2.7764e-2   |   ----   |    1.6104e-0     |  ----   |
# |      |2.500e-1   |  3.1785e-3   |  3.1268  |    6.3133e-1     |  1.3509 |   
# |      |1.250e-1   |  2.9815e-4   |  3.4142  |    2.3095e-1     |  1.4508 |    
# |      |6.250e-2   |  3.8145e-5   |  2.9665  |    8.2525e-2     |  1.4847 |
# |      |3.125e-2   |  5.4230e-6   |  2.8143  |    2.9284e-2     |  1.4947 | 
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  3   |5.000e-1   |  9.2796e-3   |   ----   |    3.3107e-1     |   ----  |
# |      |2.500e-1   |  3.4250e-4   |  4.7599  |    6.6906e-2     |  2.3069 |      
# |      |1.250e-1   |  1.4774e-5   |  4.5350  |    1.2170e-2     |  2.4588 |       
# |      |6.250e-2   |  7.6664e-7   |  4.2684  |    2.1682e-3     |  2.4888 | 
# |      |3.125e-2   |  4.4088e-8   |  4.1201  |    3.8421e-4     |  2.4965 |   
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|         
# |  4   |5.000e-1   |  2.4916e-4   |   ----   |    6.3773e-2     |   ----  |
# |      |2.500e-1   |  4.3113e-6   |  5.8528  |    5.7582e-3     |  3.4693 |      
# |      |1.250e-1   |  1.1657e-7   |  5.2089  |    5.1473e-4     |  3.4837 |       
# |      |6.250e-2   |  3.8021e-9   |  4.9383  |    4.5654e-5     |  3.4950 |  
# |      |3.125e-2   |  1.2163e-10  |  4.9662  |    4.0398e-6     |  3.4984 | 
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  5   |5.000e-1   |  1.3549e-5   |   ----   |    6.8152e-3     |   ----  |
# |      |2.500e-1   |  2.8591e-7   |  5.5665  |    3.3249e-4     |  4.3574 |      
# |      |1.250e-1   |  4.9007e-9   |  5.8664  |    1.4968e-5     |  4.4733 |       
# |      |6.250e-2   |  7.7459e-11  |  5.9834  |    6.6462e-7     |  4.4932 | 
# |      |3.125e-2   |  1.6794e-12  |  5.5274  |    2.9411e-8     |  4.4981 |   
#  Energy error of order O(k^{q-1/2}), L^2 error of order O(q^{k+1})


# T=1, gamma=1, h=k, CG q-1 in space

# |  q   |     k     | $L^2$ norm   |  rate    |   energy norm    |  rate   |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|
# |  2   |5.000e-1   |  5.6323e-1   |   ----   |    1.6104e-0     |  ----   |
# |      |2.500e-1   |  1.5238e-1   |  1.8861  |    6.4078e-1     |  1.3295 |   
# |      |1.250e-1   |  3.8942e-2   |  1.9683  |    2.3300e-1     |  1.4595 |    
# |      |6.250e-2   |  9.7781e-3   |  1.9937  |    8.2889e-2     |  1.4911 |
# |      |3.125e-2   |  2.4452e-3   |  1.9996  |    2.9347e-2     |  1.4980 | 
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  3   |5.000e-1   |  2.1979e-2   |   ----   |    3.3032e-1     |   ----  |
# |      |2.500e-1   |  2.5286e-3   |  3.1197  |    6.6911e-2     |  2.3035 |      
# |      |1.250e-1   |  2.9962e-4   |  3.0771  |    1.2170e-2     |  2.4589 |       
# |      |6.250e-2   |  3.6708e-5   |  3.0290  |    2.1682e-3     |  2.4888 | 
# |      |3.125e-2   |  4.5613e-6   |  3.0086  |    3.8421e-4     |  2.4965 |   
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|         
# |  4   |5.000e-1   |  1.9566e-3   |   ----   |    6.3830e-2     |   ----  |
# |      |2.500e-1   |  1.2436e-4   |  3.9758  |    5.7597e-3     |  3.4702 |      
# |      |1.250e-1   |  7.7114e-6   |  4.0114  |    5.1479e-4     |  3.4839 |       
# |      |6.250e-2   |  4.8969e-7   |  3.9771  |    4.5657e-5     |  3.4951 |  
# |      |3.125e-2   |  3.0656e-8   |  3.9976  |    4.0400e-6     |  3.4984 | 
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  5   |5.000e-1   |  1.5180e-4   |   ----   |    6.8763e-3     |   ----  |
# |      |2.500e-1   |  4.3686e-6   |  5.1189  |    3.3619e-4     |  4.3542 |      
# |      |1.250e-1   |  1.2188e-7   |  5.1636  |    1.5264e-5     |  4.4611 |       
# |      |6.250e-2   |  3.8641e-9   |  4.9792  |    6.9025e-7     |  4.4669 | 
# |      |3.125e-2   |  1.4273e-10  |  4.7588  |    3.1659e-8     |  4.4464 |  



 # T=1, gamma=1, h=k, CG 2q-2 in space

# |  q   |     k     | $L^2$ norm   |  rate    |   energy norm    |  rate   |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|
# |  2   |5.000e-1   |  2.7764e-2   |   ----   |    1.6104e-0     |  ----   |
# |      |2.500e-1   |  3.1785e-3   |  3.1268  |    6.3133e-1     |  1.3509 |   
# |      |1.250e-1   |  2.9815e-4   |  3.4142  |    2.3095e-1     |  1.4508 |    
# |      |6.250e-2   |  3.8145e-5   |  2.9665  |    8.2525e-2     |  1.4847 |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  3   |5.000e-1   |  8.4203e-3   |   ----   |    3.3067e-1     |   ----  |
# |      |2.500e-1   |  2.5352e-4   |  5.0537  |    6.6901e-2     |  2.3053 |      
# |      |1.250e-1   |  7.4658e-6   |  5.0857  |    1.2169e-2     |  2.4588 |       
# |      |6.250e-2   |  2.2401e-7   |  5.0587  |    2.1682e-3     |  2.4886 |  
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|         
# |  4   |5.000e-1   |  1.5878e-4   |   ----   |    6.3765e-2     |   ----  |
# |      |2.500e-1   |  1.2464e-6   |  6.9931  |    5.7580e-3     |  3.4691 |      
# |      |1.250e-1   |  9.5992e-9   |  7.0206  |    5.1472e-4     |  3.4837 |       
# |      |6.250e-2   |  7.2384e-11  |  7.0511  |    4.5654e-5     |  3.4950 |  
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  5   |5.000e-1   |  2.1083e-6   |   ----   |    6.8143e-3     |   ----  |
# |      |2.500e-1   |  4.6653e-9   |  8.8199  |    3.3247e-4     |  4.3573 |      
# |      |1.250e-1   |  1.4565e-11  |  8.3233  |    1.4967e-5     |  4.4734 |       
# |      |6.250e-2   |  1.8621e-11  |   -----  |    6.6461e-7     |  4.4931 |  
#  Energy error of order O(k^{q-1/2}), L^2 error of order O(k^{2q-1}).  Note: any L2 error hits below 1e-10, it becomes inaccuarate :( 





 # T=1, gamma=1, h=k, CG 2q-1 in space

# |  q   |     k     | $L^2$ norm   |  rate    |   energy norm    |  rate   |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|
# |  2   |5.000e-1   |  9.4398e-2   |   ----   |    1.6125e-0     |  ----   |
# |      |2.500e-1   |  1.3508e-2   |  2.8049  |    6.3137e-1     |  1.3527 |   
# |      |1.250e-1   |  1.7750e-3   |  2.9279  |    2.3095e-1     |  1.4509 |    
# |      |6.250e-2   |  2.2554e-4   |  2.9763  |    8.2525e-2     |  1.4847 |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  3   |5.000e-1   |  2.1981e-3   |   ----   |    3.3075e-1     |   ----  |
# |      |2.500e-1   |  9.6398e-5   |  4.5111  |    6.6902e-2     |  2.3056 |      
# |      |1.250e-1   |  3.2432e-6   |  4.8935  |    1.2169e-2     |  2.4588 |       
# |      |6.250e-2   |  1.0376e-7   |  4.9661  |    2.1682e-3     |  2.4886 |  
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|         
# |  4   |5.000e-1   |  7.9749e-5   |   ----   |    6.3765e-2     |   ----  |
# |      |2.500e-1   |  6.0876e-7   |  7.0334  |    5.7580e-3     |  3.4691 |      
# |      |1.250e-1   |  4.8693e-9   |  6.9660  |    5.1472e-4     |  3.4837 |       
# |      |6.250e-2   |  3.9113e-11  |  6.9599  |    4.5654e-5     |  3.4950 |  
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  5   |5.000e-1   |  2.1077e-6   |   ----   |    6.8143e-3     |   ----  |
# |      |2.500e-1   |  4.6573e-9   |  8.8219  |    3.3247e-4     |  4.3573 |      
# |      |1.250e-1   |  6.2890e-12  |  9.5342  |    1.4967e-5     |  4.4734 |       
# |      |6.250e-2   |  1.5023e-10  |   -----  |    6.6461e-7     |  4.4931 |  

#  Energy error of order O(k^{q-1/2}), L^2 error of order O(k^{2q-1}). Not 2q.




# Computed errors |||\mathbf{Z}-\mathbf{Z}_DG||| versus 1/k (loglog scale) and q=2,3,4,5
# Delta_t=np.array([ 1/5.000e-1, 1/2.500e-1, 1/1.250e-1, 1/6.250e-2]);
# error_2= np.array([  1.6104e-0, 6.3133e-1   ,  2.3095e-1 ,8.2525e-2 ]);
# error_3= np.array([  3.3107e-1   ,6.6906e-2,  1.2170e-2,  2.1682e-3 ]);
# error_4= np.array([ 6.3773e-2  , 5.7582e-3  , 5.1473e-4 ,  4.5654e-5 ]);
# error_5= np.array([   6.8152e-3,  3.3249e-4  , 1.4968e-5  , 6.6462e-7]);
# loglog(Delta_t, error_2, label='DG 2', linestyle='-', marker='o', color='blue', linewidth=3)
# loglog(Delta_t, error_3, label='DG 3', linestyle=':', marker='p', color='red', linewidth=3)
# loglog(Delta_t, error_4, label='DG 4', linestyle='--',marker='D', color='black', linewidth=3)
# loglog(Delta_t, error_5, label='DG 5', linestyle='-.', marker='s',color='green', linewidth=3)
# title('Plot of $|||\mathbf{Z}-\mathbf{Z}_{\mathrm{DG}}|||$ versus 1/k')
# legend(loc = 'best')
# grid()
# savefig('1D_linear_wave_energy_error.png', dpi=300, bbox_inches='tight')


# clf()
# # Computed errors ||dot u- dot u_DG|| versus 1/k (loglog scale) and q=2,3,4,5
# Delta_t=np.array([ 1/5.000e-1, 1/2.500e-1, 1/1.250e-1, 1/6.250e-2]);
# error_2= np.array([ 2.7764e-2, 3.1785e-3, 2.9815e-4, 3.8145e-5]);
# error_3= np.array([ 9.2796e-3, 3.4250e-4, 1.4774e-5, 7.6664e-7]);
# error_4= np.array([ 2.4916e-4, 4.3113e-6, 1.1657e-7, 3.8021e-9]);
# error_5= np.array([ 1.3549e-5, 2.8591e-7, 4.9007e-9, 7.7459e-11]);
# loglog(Delta_t, error_2, label='DG 2', linestyle='-', marker='o', color='blue', linewidth=3)
# loglog(Delta_t, error_3, label='DG 3', linestyle=':', marker='p', color='red', linewidth=3)
# loglog(Delta_t, error_4, label='DG 4', linestyle='--',marker='D', color='black', linewidth=3)
# loglog(Delta_t, error_5, label='DG 5', linestyle='-.', marker='s',color='green', linewidth=3)
# title('Plot of $||\dot{u}- \dot{u}_\mathrm{DG}||_{L^2(\Omega)}$ versus 1/k')
# legend(loc = 'best')
# grid()
# savefig('1D_linear_wave_l2_error.png', dpi=300, bbox_inches='tight')


# Computational times for the time loop 
# |  p   |     k     |  1 time step |  total   | L2 norm (v) |  rate  | 
# |------|:---------:|:------------:|:--------:|:-----------:|:------:|
# |  2   |5.00e-1    |  4.10e-2s    | 2.49e-1s |  1.2283e-0  |  ----  | 
# |      |2.50e-1    |  3.94e-2s    | 3.28e-1s |  5.4672e-1  | 1.1678 |                        
# |      |1.25e-1    |  5.82e-2s    | 7.06e-1s |  1.5308e-1  | 1.8365 | 
# |      |6.25e-2    |  1.14e-1s    | 2.24e-0s |  3.9282e-2  | 1.9623 | 




#27/10/2022 Numerical tests again

# T=1, gamma=0, h=k, CG q in space

# |  q   |     k     | $L^2$ norm   |  rate    |   energy norm    |  rate   |
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|
# |  2   |5.000e-1   |  7.9630e-2   |   ----   |    1.5990e-0     |  ----   |
# |      |2.500e-1   |  1.1466e-2   |  2.7959  |    6.3820e-1     |  1.3251 |   
# |      |1.250e-1   |  1.3812e-3   |  3.0534  |    2.3267e-1     |  1.4557 |    
# |      |6.250e-2   |  1.6587e-4   |  3.0578  |    8.2854e-2     |  1.4896 | 
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  3   |5.000e-1   |  5.0427e-3   |   ----   |    3.3338e-1     |   ----  |
# |      |2.500e-1   |  1.6240e-4   |  4.9566  |    6.7318e-2     |  2.3081 |      
# |      |1.250e-1   |  6.7536e-6   |  4.5878  |    1.2210e-2     |  2.4629 |       
# |      |6.250e-2   |  3.5833e-7   |  4.2363  |    2.1720e-3     |  2.4910 |   
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|         
# |  4   |5.000e-1   |  1.1686e-4   |   ----   |    6.4121e-2     |   ----  |
# |      |2.500e-1   |  3.3180e-6   |  5.1383  |    5.7781e-3     |  3.4721 |      
# |      |1.250e-1   |  1.0242e-7   |  5.0177  |    5.1568e-4     |  3.4860 |       
# |      |6.250e-2   |  3.1876e-9   |  5.0059  |    4.5698e-5     |  3.4963 |  
# |------|:---------:|:------------:|:--------:|:----------------:|:-------:|            
# |  5   |5.000e-1   |  6.8209e-6   |   ----   |    6.8366e-3     |   ----  |
# |      |2.500e-1   |  1.0444e-7   |  6.0292  |    3.3321e-4     |  4.3588 |      
# |      |1.250e-1   |  1.6308e-9   |  6.0010  |    1.4985e-5     |  4.4748 |       
# |      |6.250e-2   |  2.5481e-11  |  6.0000  |    6.6502e-7     |  4.4940 | 

#  Energy error of order O(k^{q-1/2}), L^2 error of order O(q^{k+1})
