from fenics import *
import numpy as np
from matplotlib.pyplot import*
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
from scipy.integrate import quad
from numpy import linalg as LA

# We consider the 1D nonlinear problem
# u_tt+2*gamma*u_t+gamma^2*u-(S(u_x))_x =f with S(u_x)=1/3 (u_x)^3.
# u(0,t)=u(1,t)=0  for all t in (0,T] with T=1.
# The exact solution in this case is u=sin(sqrt(2)*pi*t)sin(pi*x) such that
#  u(0,x)=0,  u_t(0,x)=sqrt(2)*pi*sin(pi*x) and
# f= [(-2*pi^2+gamma^2)*sin(sqrt(2)*pi*t)+2*sqrt(2)*gamma*pi*cos(sqrt(2)*pi*t)]sin(p*x)+pi^4sin^3(sqrt(2)*pi*t)cos^2(pi*x)*sin(pi*x).


# Create mesh and define function space
nx=2               # Number of subintervals in the spatial domain
tmax    = 1        # Max time 
xmax  = 1          # Max point of x domain 
h= xmax/(nx)       # calculate the uniform mesh-size 
k= h**2            # time step size 
nt=int(tmax/k+1)   # initialize time axis
time= np.arange(0, nt) * k
print('The spatial mesh size is', h)
print('The time step is', k, 'with', nt, 'time steps')

mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 4)
V_vec = VectorFunctionSpace(mesh, "CG", 4)

#  Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0.0), boundary)

u_D=      Expression("sin(sqrt(2)*pi*t)*sin(pi*x[0])", degree=10, t=0)
u_Dprime= Expression("sqrt(2)*pi*cos(sqrt(2)*pi*t)*sin(pi*x[0])", degree=10,t=0)

f_space1= Expression("sin(pi*x[0])", degree=10)
f_space4= Expression("pow(cos(pi*x[0]),2)*sin(pi*x[0])",degree=10)

u = TrialFunction(V)
v = TestFunction(V)
A1=Expression("pi*cos(pi*x[0])",degree=10)
A2=Constant(0.0)
A3=Constant(0.0)
A4=Constant(0.0)
A5=Constant(0.0)


An_1= interpolate(A1, V)
An_2= interpolate(A2, V)
An_3= interpolate(A3, V)
An_4= interpolate(A4, V)
An_5= interpolate(A5, V)


# Mass form
m= dot(u,v)*dx

# RHS
f1= dot(v, f_space1)*dx
f4= dot(v, f_space4)*dx

# Assemble matrices and RHS vectors
M_bar, F1= assemble_system(m, f1, bc)                                                                          
_, F4= assemble_system(m, f4,bc)                                                                          
M_array=M_bar.array()
F1_array= F1.get_local()
F4_array= F4.get_local()

# stiffness forms
k11= dot(pow(An_1,2)*grad(u),grad(v))*dx 
k22= dot(pow(An_2,2)*grad(u),grad(v))*dx 
k33= dot(pow(An_3,2)*grad(u),grad(v))*dx 
k44= dot(pow(An_4,2)*grad(u),grad(v))*dx
k55= dot(pow(An_5,2)*grad(u),grad(v))*dx
k12= dot(An_1*An_2*grad(u),grad(v))*dx 
k13= dot(An_1*An_3*grad(u),grad(v))*dx 
k14= dot(An_1*An_4*grad(u),grad(v))*dx
k15= dot(An_1*An_5*grad(u),grad(v))*dx
k23= dot(An_2*An_3*grad(u),grad(v))*dx 
k24= dot(An_2*An_4*grad(u),grad(v))*dx 
k25= dot(An_2*An_5*grad(u),grad(v))*dx 
k34= dot(An_3*An_4*grad(u),grad(v))*dx 
k35= dot(An_3*An_5*grad(u),grad(v))*dx
k45= dot(An_4*An_5*grad(u),grad(v))*dx
                                                                         
K11,_ = assemble_system(k11, f1, bc)                                                                          
K22,_ = assemble_system(k22, f1, bc) 
K33,_ = assemble_system(k33, f1, bc) 
K44,_ = assemble_system(k44, f1, bc)
K55,_ = assemble_system(k55, f1, bc)
K12,_ = assemble_system(k12, f1, bc) 
K13,_ = assemble_system(k13, f1, bc)  
K14,_ = assemble_system(k14, f1, bc) 
K15,_ = assemble_system(k15, f1, bc)
K23,_ = assemble_system(k23, f1, bc) 
K24,_ = assemble_system(k24, f1, bc)
K25,_ = assemble_system(k25, f1, bc) 
K34,_ = assemble_system(k34, f1, bc) 
K35,_ = assemble_system(k35, f1, bc)  
K45,_ = assemble_system(k45, f1, bc) 

K11_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K11.array() @fractional_matrix_power(M_array,-1/2))                                     
K22_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K22.array() @fractional_matrix_power(M_array,-1/2)) 
K33_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K33.array() @fractional_matrix_power(M_array,-1/2)) 
K44_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K44.array() @fractional_matrix_power(M_array,-1/2)) 
K55_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K55.array() @fractional_matrix_power(M_array,-1/2))
K12_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K12.array() @fractional_matrix_power(M_array,-1/2))                                       
K13_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K13.array() @fractional_matrix_power(M_array,-1/2)) 
K14_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K14.array() @fractional_matrix_power(M_array,-1/2)) 
K15_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K15.array() @fractional_matrix_power(M_array,-1/2)) 
K23_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K23.array() @fractional_matrix_power(M_array,-1/2)) 
K24_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K24.array() @fractional_matrix_power(M_array,-1/2)) 
K25_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K25.array() @fractional_matrix_power(M_array,-1/2)) 
K34_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K34.array() @fractional_matrix_power(M_array,-1/2)) 
K35_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K35.array() @fractional_matrix_power(M_array,-1/2)) 
K45_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K45.array() @fractional_matrix_power(M_array,-1/2)) 

u_space= Expression("sin(pi*x[0])", degree=10)
u_e = interpolate(u_space,V).vector().get_local()


gamma=1
d=np.size(u_e)
print('The dimension of the vector is', d)


# Construct the relevant matrices for the resulting ODE systems
Z0=np.zeros(d);
Z1=np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2) @u_e
L= 2*gamma*np.identity(d)
K0= gamma**2*np.identity(d)

## Exact solutions 
Z_e = lambda  t: fractional_matrix_power(M_array,1/2)@u_e*np.sin(np.sqrt(2)*np.pi*t)
Zprime_e = lambda  t: np.sqrt(2)*np.pi*fractional_matrix_power(M_array,1/2)@u_e*np.cos(np.sqrt(2)*np.pi*t)


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
M4[1,1]=4/(k**2); M4[1,2]=-12/(k**2); M4[1,3]=24/(k**2);M4[1,4]=-40/(k**2);
M4[2,1]=-12/(k**2); M4[2,2]=36/(k**2); M4[2,3]=-72/(k**2); M4[2,4]=120/(k**2);
M4[3,1]=24/(k**2); M4[3,2]=-72/(k**2); M4[3,3]=144/(k**2); M4[3,4]=-240/(k**2);
M4[4,1]=-40/(k**2); M4[4,2]=120/(k**2); M4[4,3]=-240/(k**2); M4[4,4]=400/(k**2);


M5=np.zeros((Dt,Dt),dtype=float); 
M5[0,0]=1; M5[0,1]=-1; M5[0,2]=1; M5[0,3]=-1; M5[0,4]=1;
M5[1,0]=-1; M5[1,1]=1; M5[1,2]=-1; M5[1,3]=1; M5[1,4]=-1;
M5[2,0]=1; M5[2,1]=-1; M5[2,2]=1; M5[2,3]=-1; M5[2,4]=1;
M5[3,0]=-1; M5[3,1]=1; M5[3,2]=-1; M5[3,3]=1; M5[3,4]=-1;
M5[4,0]= 1; M5[4,1]=-1; M5[4,2]=1; M5[4,3]=-1; M5[4,4]=1;

M=M1+M4


#Initialisation
Z_minus=np.zeros([nt,d])           #intialize  Z(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
Z_plus= np.zeros([nt,d])           #intialize  Z(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
Zprime_minus= np.zeros([nt,d])     #intialize  Z'(t_n^{-}, x_m) for n=0,1,...,N, m=1, 2,...,d
Zprime_plus = np.zeros([nt,d])     #intialize  Z'(t_n^{+}, x_m) for n=0,1,...,N, m=1, 2,...,d
Z_minus[0,:]=Z0
Zprime_minus[0,:]=Z1
b=np.zeros(Dt*d)
np.shape(Z_minus)
print(np.size(b))



#Iterations DG4
t=0
error_energy=0;
sint1= lambda t : np.sin(np.sqrt(2)*np.pi*t)*2/k;
cost1= lambda t : np.cos(np.sqrt(2)*np.pi*t)*2/k;
sint1_3= lambda t : pow(np.sin(np.sqrt(2)*np.pi*t),3)*2/k;
for it in range(1,nt):
    t+=k
    u_D.t=t
    u_Dprime.t=t
    print(it)
    def phi(t,i):
        if i==0:
            return 1;
        if i==1:
            return 2*(t-time[it-1])/k -1;
        if i==2:
            return 6*(t-time[it-1])**2/(k**2)-6*(t-time[it-1])/k +1;
        if i==3:
            return 20*(t-time[it-1])**3/(k**3)-30*(t-time[it-1])**2/k**2+12*(t-time[it-1])/k-1;
        if i==4:
            return 70*(t-time[it-1])**4/k**4-140*(t-time[it-1])**3/k**3+90*(t-time[it-1])**2/k**2-20*(t-time[it-1])/k+1;

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

    def int_t(t, i,j, m,n):
        if m==n:
            return phi_t(t,i)*phi(t,j)*(phi(t,m)**2)
        else:
            return 2*phi_t(t,i)*phi(t,j)*(phi(t,m)*phi(t,n))

    def delta(i,j):
        if i== j:
            return 1
        else:
            return 0
    
    M3_bar=[[[[quad(lambda t: int_t(t,i,j,m,n), time[it-1], time[it])[0] for j in range (0,5)] for i in range(0,5)] for n in range(0,5) ] for m in range(0,5)]
    M5_bar=[[ 2*(phi(time[it-1],i)*phi(time[it-1],j))*M5*(1-delta(i,j))+ (phi(time[it-1],i)**2)*delta(i,j)*M5 for j in range(0,5)] for i in range(0,5)]

    
    Gsin=(-2*(np.pi**2)+gamma**2)*(fractional_matrix_power(M_array,-1/2)@ F1_array)
    Gcos=(2*np.sqrt(2)*gamma*np.pi)*(fractional_matrix_power(M_array,-1/2)@ F1_array)
    Gsin3=pow(np.pi,4)*(fractional_matrix_power(M_array,-1/2)@ F4_array)

    
    sint2= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    cost2= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,2);
    sint2_3= lambda t : pow(np.sin(np.sqrt(2)*np.pi*t),3)*phi_t(t,2);
    sint3= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,3);
    cost3= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,3);
    sint3_3= lambda t :  pow(np.sin(np.sqrt(2)*np.pi*t),3)*phi_t(t,3);
    sint4= lambda t : np.sin(np.sqrt(2)*np.pi*t)*phi_t(t,4);
    cost4= lambda t : np.cos(np.sqrt(2)*np.pi*t)*phi_t(t,4);
    sint4_3= lambda t :  pow(np.sin(np.sqrt(2)*np.pi*t),3)*phi_t(t,4)



     #Compute the block diagonal matrix A1
    A_1=M   
    for i in range(1,d):
        A_1= block_diag(A_1,M)    
     #Compute the block matrix B
    B=np.block([[L[i, j]*M2+K0[i,j]*(M3+M5)\
      +K11_bar[i,j]*(M3_bar[0][0]+M5_bar[0][0])+K22_bar[i,j]*(M3_bar[1][1]+M5_bar[1][1])+K33_bar[i,j]*(M3_bar[2][2]+M5_bar[2][2])+K44_bar[i,j]*(M3_bar[3][3]+M5_bar[3][3])+K55_bar[i,j]*(M3_bar[4][4]+M5_bar[4][4])\
      +K12_bar[i,j]*(M3_bar[0][1]+M5_bar[0][1])+K13_bar[i,j]*(M3_bar[0][2]+M5_bar[0][2])+K14_bar[i,j]*(M3_bar[0][3]+M5_bar[0][3])+K15_bar[i,j]*(M3_bar[0][4]+M5_bar[0][4])\
      +K23_bar[i,j]*(M3_bar[1][2]+M5_bar[1][2])+K24_bar[i,j]*(M3_bar[1][3]+M5_bar[1][3])+K25_bar[i,j]*(M3_bar[1][4]+M5_bar[1][4])\
      +K34_bar[i,j]*(M3_bar[2][3]+M5_bar[2][3])+K35_bar[i,j]*(M3_bar[2][4]+M5_bar[2][4])+K45_bar[i,j]*(M3_bar[3][4]+M5_bar[3][4]) for j in range(0,d) ] for i in range(0,d)])
    A=A_1+B; 
    
    b[0::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])
            
    b[1::Dt]=-((K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:]))\
    +Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos+quad(sint1_3, time[it-1], time[it])[0]*Gsin3
    b[2::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])\
    +Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos+quad(sint2_3, time[it-1], time[it])[0]*Gsin3
    b[3::Dt]= -((K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:]))\
    +Zprime_minus[it-1,:]*(12/k)+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos+quad(sint3_3, time[it-1], time[it])[0]*Gsin3
    b[4::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])\
     +Zprime_minus[it-1,:]*(-20/k)+quad(sint4, time[it-1], time[it])[0]*Gsin+quad(cost4, time[it-1], time[it])[0]*Gcos+quad(sint4_3, time[it-1], time[it])[0]*Gsin3
    eps=1.0
    tol=1e-10
    maxiter= 30
    iter =0
    while eps>tol and iter<maxiter:
        iter +=1   
        alpha=np.linalg.solve(A,b)

        U_DG_1= fractional_matrix_power(M_array,-1/2)@ alpha[0::Dt]
        U_DG_2= fractional_matrix_power(M_array,-1/2)@ alpha[1::Dt]
        U_DG_3= fractional_matrix_power(M_array,-1/2)@ alpha[2::Dt]
        U_DG_4= fractional_matrix_power(M_array,-1/2)@ alpha[3::Dt]
        U_DG_5= fractional_matrix_power(M_array,-1/2)@ alpha[4::Dt]

        # u_DG_1= Function(V)
        # u_DG_1.vector().set_local(U_DG_1)
        # grad_Un_1= Function(V)
        # grad_Un_1.vector()[:]=project(u_DG_1.dx(0),V).vector()
        
        # u_DG_2= Function(V)
        # u_DG_2.vector().set_local(U_DG_2)
        # grad_Un_2= Function(V)
        # grad_Un_2.vector()[:]=project(u_DG_2.dx(0),V).vector()
    
        # u_DG_3= Function(V)
        # u_DG_3.vector().set_local(U_DG_3)
        # grad_Un_3= Function(V)
        # grad_Un_3.vector()[:]=project(u_DG_3.dx(0),V).vector() 

        # u_DG_4= Function(V)
        # u_DG_4.vector().set_local(U_DG_4)
        # grad_Un_4= Function(V)
        # grad_Un_4.vector()[:]=project(u_DG_4.dx(0),V).vector()

        # u_DG_5= Function(V)
        # u_DG_5.vector().set_local(U_DG_5)
        # grad_Un_5= Function(V)
        # grad_Un_5.vector()[:]=project(u_DG_5.dx(0),V).vector()


        u_DG_1= Function(V)
        u_DG_1.vector().set_local(U_DG_1)
        grad_Un_1= Function(V)
        grad_Un_1.vector().set_local(project(grad(u_DG_1),V_vec).vector())

        u_DG_2= Function(V)
        u_DG_2.vector().set_local(U_DG_2)
        grad_Un_2= Function(V)
        grad_Un_2.vector().set_local(project(grad(u_DG_2),V_vec).vector())
        
        u_DG_3= Function(V)
        u_DG_3.vector().set_local(U_DG_3)
        grad_Un_3= Function(V)
        grad_Un_3.vector().set_local(project(grad(u_DG_3),V_vec).vector())

        u_DG_4= Function(V)
        u_DG_4.vector().set_local(U_DG_4)
        grad_Un_4= Function(V)
        grad_Un_4.vector().set_local(project(grad(u_DG_4),V_vec).vector())
        
        u_DG_5= Function(V)
        u_DG_5.vector().set_local(U_DG_5)
        grad_Un_5= Function(V)
        grad_Un_5.vector().set_local(project(grad(u_DG_5),V_vec).vector())
    
        diff1= grad_Un_1.vector()-An_1.vector()
        diff2= grad_Un_2.vector()-An_2.vector()
        diff3= grad_Un_3.vector()-An_3.vector()
        diff4= grad_Un_4.vector()-An_4.vector()
        diff5= grad_Un_5.vector()-An_5.vector()
        eps= LA.norm(diff1, ord=np.Inf)+ LA.norm(diff2, ord=np.Inf)+ LA.norm(diff3, ord=np.Inf)+LA.norm(diff4, ord=np.Inf)+LA.norm(diff5, ord=np.Inf)
        print('iter= % d: norm =%g' %(iter, eps))

        An_1.assign(grad_Un_1)
        An_2.assign(grad_Un_2)
        An_3.assign(grad_Un_3)
        An_4.assign(grad_Un_4)
        An_5.assign(grad_Un_5)

        # stiffness forms
        k11= dot(pow(An_1,2)*grad(u),grad(v))*dx 
        k22= dot(pow(An_2,2)*grad(u),grad(v))*dx 
        k33= dot(pow(An_3,2)*grad(u),grad(v))*dx 
        k44= dot(pow(An_4,2)*grad(u),grad(v))*dx
        k55= dot(pow(An_5,2)*grad(u),grad(v))*dx
        k12= dot(An_1*An_2*grad(u),grad(v))*dx 
        k13= dot(An_1*An_3*grad(u),grad(v))*dx 
        k14= dot(An_1*An_4*grad(u),grad(v))*dx
        k15= dot(An_1*An_5*grad(u),grad(v))*dx
        k23= dot(An_2*An_3*grad(u),grad(v))*dx 
        k24= dot(An_2*An_4*grad(u),grad(v))*dx 
        k25= dot(An_2*An_5*grad(u),grad(v))*dx 
        k34= dot(An_3*An_4*grad(u),grad(v))*dx 
        k35= dot(An_3*An_5*grad(u),grad(v))*dx
        k45= dot(An_4*An_5*grad(u),grad(v))*dx
        K11,_ = assemble_system(k11, f1, bc)                                                                          
        K22,_ = assemble_system(k22, f1, bc) 
        K33,_ = assemble_system(k33, f1, bc) 
        K44,_ = assemble_system(k44, f1, bc)
        K55,_ = assemble_system(k55, f1, bc)
        K12,_ = assemble_system(k12, f1, bc) 
        K13,_ = assemble_system(k13, f1, bc)  
        K14,_ = assemble_system(k14, f1, bc) 
        K15,_ = assemble_system(k15, f1, bc)
        K23,_ = assemble_system(k23, f1, bc) 
        K24,_ = assemble_system(k24, f1, bc)
        K25,_ = assemble_system(k25, f1, bc)
        K34,_ = assemble_system(k34, f1, bc) 
        K35,_ = assemble_system(k35, f1, bc)  
        K45,_ = assemble_system(k45, f1, bc) 
        K11_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K11.array() @fractional_matrix_power(M_array,-1/2))                                     
        K22_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K22.array() @fractional_matrix_power(M_array,-1/2)) 
        K33_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K33.array() @fractional_matrix_power(M_array,-1/2)) 
        K44_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K44.array() @fractional_matrix_power(M_array,-1/2)) 
        K55_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K55.array() @fractional_matrix_power(M_array,-1/2))
        K12_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K12.array() @fractional_matrix_power(M_array,-1/2))                                       
        K13_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K13.array() @fractional_matrix_power(M_array,-1/2)) 
        K14_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K14.array() @fractional_matrix_power(M_array,-1/2)) 
        K15_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K15.array() @fractional_matrix_power(M_array,-1/2)) 
        K23_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K23.array() @fractional_matrix_power(M_array,-1/2)) 
        K24_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K24.array() @fractional_matrix_power(M_array,-1/2)) 
        K25_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K25.array() @fractional_matrix_power(M_array,-1/2)) 
        K34_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K34.array() @fractional_matrix_power(M_array,-1/2)) 
        K35_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K35.array() @fractional_matrix_power(M_array,-1/2)) 
        K45_bar=(1/3)*fractional_matrix_power(M_array,-1/2)@ (K45.array() @fractional_matrix_power(M_array,-1/2)) 
        
        #Compute the block diagonal matrix A1

        A_1=M   
        for i in range(1,d):
            A_1= block_diag(A_1,M)    
        #Compute the block matrix B
        B=np.block([[L[i, j]*M2+K0[i,j]*(M3+M5)\
        +K11_bar[i,j]*(M3_bar[0][0]+M5_bar[0][0])+K22_bar[i,j]*(M3_bar[1][1]+M5_bar[1][1])+K33_bar[i,j]*(M3_bar[2][2]+M5_bar[2][2])+K44_bar[i,j]*(M3_bar[3][3]+M5_bar[3][3])+K55_bar[i,j]*(M3_bar[4][4]+M5_bar[4][4])\
        +K12_bar[i,j]*(M3_bar[0][1]+M5_bar[0][1])+K13_bar[i,j]*(M3_bar[0][2]+M5_bar[0][2])+K14_bar[i,j]*(M3_bar[0][3]+M5_bar[0][3])+K15_bar[i,j]*(M3_bar[0][4]+M5_bar[0][4])\
        +K23_bar[i,j]*(M3_bar[1][2]+M5_bar[1][2])+K24_bar[i,j]*(M3_bar[1][3]+M5_bar[1][3])+K25_bar[i,j]*(M3_bar[1][4]+M5_bar[1][4])\
        +K34_bar[i,j]*(M3_bar[2][3]+M5_bar[2][3])+K35_bar[i,j]*(M3_bar[2][4]+M5_bar[2][4])+K45_bar[i,j]*(M3_bar[3][4]+M5_bar[3][4]) for j in range(0,d) ] for i in range(0,d)])
        A=A_1+B; 
        b[0::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])
            
        b[1::Dt]=-((K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:]))\
        +Zprime_minus[it-1,:]*2/k+quad(sint1, time[it-1], time[it])[0]*Gsin+quad(cost1, time[it-1], time[it])[0]*Gcos+quad(sint1_3, time[it-1], time[it])[0]*Gsin3

        b[2::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])\
        +Zprime_minus[it-1,:]*(-6/k)+quad(sint2, time[it-1], time[it])[0]*Gsin+quad(cost2, time[it-1], time[it])[0]*Gcos+quad(sint2_3, time[it-1], time[it])[0]*Gsin3

        b[3::Dt]= -((K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:]))\
        +Zprime_minus[it-1,:]*(12/k)+quad(sint3, time[it-1], time[it])[0]*Gsin+quad(cost3, time[it-1], time[it])[0]*Gcos+quad(sint3_3, time[it-1], time[it])[0]*Gsin3

        b[4::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])+(phi(time[it-1],3)**2)*(K44_bar@Z_minus[it-1,:])+(phi(time[it-1],4)**2)*(K55_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@Z_minus[it-1,:])+(2*phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@Z_minus[it-1,:])+(2*phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@Z_minus[it-1,:])\
        +Zprime_minus[it-1,:]*(-20/k)+quad(sint4, time[it-1], time[it])[0]*Gsin+quad(cost4, time[it-1], time[it])[0]*Gcos+quad(sint4_3, time[it-1], time[it])[0]*Gsin3
    
    convergence = 'convergence after %d Picard iterations' % iter
    if iter>= maxiter:
        convergence = ' no' + convergence
    print ('''
        Solution of the nonlinear elastodynamics problemu_tt+2*gamma*u_t+gamma^2*u-(S(u_x))_x =f with S(u_x)=1/3 (u_x)^3,
        with f= [(-2*pi^2+gamma^2)*sin(sqrt(2)*pi*t)+2*sqrt(2)*gamma*pi*cos(sqrt(2)*pi*t)]sin(pi*x)+2pi^4sin^3(sqrt(2)*pi*t)cos^2(pi*x)sin(pi*x),
        u(0,x)=0, u_t(0,x)= sqrt(2)*pi*sin(pi*x) and u=0 at x=0 and  x=1.
        %s
        ''' % ( convergence))
    integrand=lambda t: 2*gamma*np.dot(alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-Zprime_e(t), alpha[1::Dt]*phi_t(t,1)+alpha[2::Dt]*phi_t(t,2)+alpha[3::Dt]*phi_t(t,3)+alpha[4::Dt]*phi_t(t,4)-Zprime_e(t))
    Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]+alpha[3::Dt]+alpha[4::Dt]
    Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]-alpha[3::Dt]+alpha[4::Dt]
    Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k+alpha[3::Dt]*12/k+alpha[4::Dt]*(20/k)
    Zprime_plus[it-1,:]=alpha[1::Dt]*2/k-alpha[2::Dt]*6/k+alpha[3::Dt]*12/k-alpha[4::Dt]*(20/k)
    error_energy=1/2*(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])@(Zprime_plus[it-1,:]-Zprime_minus[it-1,:])+1/2*(K0@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])\
                 +1/2*(phi(time[it-1],0)**2)*(K11_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+1/2*(phi(time[it-1],1)**2)*(K22_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+1/2*(phi(time[it-1],2)**2)*(K33_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+1/2*(phi(time[it-1],3)**2)*(K44_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+1/2*(phi(time[it-1],4)**2)*(K55_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])\
                 +(phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],0)*phi(time[it-1],2))*(K13_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],0)*phi(time[it-1],3))*(K14_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],0)*phi(time[it-1],4))*(K15_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])\
                 +(phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],1)*phi(time[it-1],3))*(K24_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],1)*phi(time[it-1],4))*(K25_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])\
                 +(phi(time[it-1],2)*phi(time[it-1],3))*(K34_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],2)*phi(time[it-1],4))*(K35_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+(phi(time[it-1],3)*phi(time[it-1],4))*(K45_bar@(Z_plus[it-1,:]-Z_minus[it-1,:]))@(Z_plus[it-1,:]-Z_minus[it-1,:])+quad(integrand,time[it-1], time[it])[0] +error_energy
    print('the error in energy norm is', np.sqrt(error_energy))

energy=error_energy+1/2*(Zprime_minus[nt-1,:]-Zprime_e(time[nt-1]))@(Zprime_minus[nt-1,:]-Zprime_e(time[nt-1]))+1/2*(K0@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))\
+1/2*(phi(time[nt-1],0)**2)*(K11_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+1/2*(phi(time[nt-1],1)**2)*(K22_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+1/2*(phi(time[nt-1],2)**2)*(K33_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+1/2*(phi(time[nt-1],3)**2)*(K44_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+1/2*(phi(time[nt-1],4)**2)*(K55_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))\
+(phi(time[nt-1],0)*phi(time[nt-1],1))*(K12_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],0)*phi(time[nt-1],2))*(K13_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],0)*phi(time[nt-1],3))*(K14_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],0)*phi(time[it-1],4))*(K15_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))\
+(phi(time[nt-1],1)*phi(time[nt-1],2))*(K23_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],1)*phi(time[nt-1],3))*(K24_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],1)*phi(time[nt-1],4))*(K25_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))\
+(phi(time[nt-1],2)*phi(time[it-1],3))*(K34_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],2)*phi(time[nt-1],4))*(K35_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))+(phi(time[nt-1],3)*phi(time[nt-1],4))*(K45_bar@(Z_minus[nt-1,:]-Z_e(time[nt-1])))@(Z_minus[nt-1,:]-Z_e(time[nt-1]))
print('the energy error norm is',np.sqrt(energy))
print('the energy error norm without last step is', np.sqrt(error_energy))


# Compute the error in L2 norm
U_end_DG= fractional_matrix_power(M_array,-1/2)@ Z_minus[nt-1,:]
Uprime_end_DG= fractional_matrix_power(M_array,-1/2)@ Zprime_minus[nt-1,:]

u_end_DG= Function(V)
# u_end_DG.vector().set_local(U_end_DG)
u_end_DG.vector()[:]=U_end_DG

uprime_end_DG= Function(V)
# uprime_end_DG.vector().set_local(Uprime_end_DG)
uprime_end_DG.vector()[:]=Uprime_end_DG

error1=errornorm(u_D, u_end_DG, 'L2')
error2=errornorm(u_Dprime, uprime_end_DG, 'L2')
error=error1+error2
print('The L2 error ||u-u_DG||+||dot u- dot u_DG|| is', error )
print('The L2 error ||dot u- dot u_DG|| is', error2 )


# T=1, gamma=1 tol=1e-10  CG p in space k=h^2, MaxIteration=30

# | p=q  |   h       |    k=h^2     |  L^2 norm  |  rate  | 
# |------|:---------:|:------------:|:----------:|: -----:|
# |  2   | 2.500e-2  |  6.25000e-2  |  1.2123e-2 |  ----  |    
# |      | 2.000e-1  |  4.00000e-2  |  4.9774e-3 | 1.9948 |  
# |      | 1.250e-1  |  1.56250e-2  |  1.1643e-3 | 1.5455 |   
# |      | 6.250e-2  |  3.90625e-3  |  1.2454e-4 | 1.6124 | 
# |------|:---------:|:------------:|:----------:|:------:|
# |  3   | 2.500e-2  |  6.25000e-2  |  4.1533e-4 |  ----  |    
# |      | 2.000e-1  |  4.00000e-2  |  1.8590e-4 | 1.8012 |  
# |      | 1.250e-1  |  1.56250e-2  |  2.5283e-5 | 2.1224 |   
# |      | 6.250e-2  |  3.90625e-3  |  1.5609e-6 | 2.0089 | 
# |------|:---------:|:------------:|:----------:|:------:|
# |  4   | 2.500e-2  |  6.25000e-2  |  2.5498e-5 |  ----  |    
# |      | 2.000e-1  |  4.00000e-2  |  8.3999e-6 | 2.4881 |  
# |      | 1.250e-1  |  1.56250e-2  |  9.7790e-7 | 2.2878 |   
# |      | 6.250e-2  |  3.90625e-3  |  3.1115e-8 | 2.4870 |



clf()
# Computed errors ||dot u- dot u_DG||+||u-u_DG|| versus 1/k (loglog scale) and q=2,3,4
Delta_t= np.array([ 1/6.25000e-2, 1/4.00000e-2, 1/1.56250e-2, 1/3.90625e-3]);
error_2= np.array([ 1.2123e-2 , 4.9774e-3  , 1.1643e-3 , 1.2454e-4 ]);
error_3= np.array([ 4.1533e-4 , 1.8590e-4  , 2.5283e-5 , 1.5609e-6 ]);
error_4= np.array([ 2.5498e-5 , 8.3999e-6  , 9.7790e-7 , 3.1115e-8 ]);
loglog(Delta_t, error_2, label='DG 2', linestyle='-', marker='o', color='blue', linewidth=3)
loglog(Delta_t, error_3, label='DG 3', linestyle=':', marker='p', color='red', linewidth=3)
loglog(Delta_t, error_4, label='DG 4', linestyle='--',marker='D', color='black', linewidth=3)
title('Plot of $||\dot{u}- \dot{u}_\mathrm{DG}||_{L^2(\Omega)}$+$||u- u_\mathrm{DG}||_{L^2(\Omega)}$ versus 1/k')
legend(loc = 'best')
grid()
savefig('1D_nonlinear_elastodynamics_l2_error.png', dpi=300, bbox_inches='tight')
