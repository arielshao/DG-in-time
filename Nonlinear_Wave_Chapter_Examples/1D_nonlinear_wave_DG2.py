from fenics import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag
from scipy.integrate import quad
from numpy import linalg as LA
from matplotlib.pyplot import *

# We consider the 1D nonlinear wave problem
# u_{tt}+a'(u)u_{t}-(b(u))_{xx} =f with a(u)=u+u^3, b(u)=u+u^3
# The exact solution in this case is u=exp(t)*sin(pi*x), u(0,x)=sin(pi*x) and u_t(0,x)=sin(pi*x)
# f=(pi^2+2)*exp(t)*sin(pi*x)+(3pi^2+3)*exp(3*t)*sin^3(pi*x)-6pi^2*exp(3*t)*cos^2(pi*x)*sin(pi*x)


# Create mesh and define function space
nx=12                  # Number of subintervals in the spatial domain
tmax    = 1            # Max time 
xmax  = 1              # Max point of x domain 
h= xmax/(nx)           # calculate the uniform mesh-size 
k=h**2                 # time step size 
nt=int(tmax/k+1)       # initialize time axis
time= np.arange(0, nt) * k
print('The spatial mesh size is', h)
print('The time step is',k, 'with', nt, 'time steps')

mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, "CG", 5)

#  Define boundary condition
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0.0), boundary)

u_D=      Expression("exp(t)*sin(pi*x[0])", degree=10, t=0)
u_Dprime= Expression("exp(t)*sin(pi*x[0])", degree=10, t=0)

f_space1= Expression("sin(pi*x[0])", degree=10)
f_space2= Expression("pow(sin(pi*x[0]),3)", degree=10)
f_space3= Expression("pow(cos(pi*x[0]),2)*sin(pi*x[0])",degree=10)

u = TrialFunction(V)
v = TestFunction(V)
A1=Expression("sin(pi*x[0])",degree=10)
A2=Constant(0.0)
A3=Constant(0.0)

An_1= interpolate(A1, V);
An_2= interpolate(A2, V);
An_3= interpolate(A3, V);

# Mass form
m= dot(u,v)*dx

# RHS
f1= dot(v, f_space1)*dx
f2= dot(v, f_space2)*dx
f3= dot(v, f_space3)*dx

# Stiffness form 
k0= dot(grad(u),grad(v))*dx 

# Assemble matrices and RHS vectors
M_bar, F1= assemble_system(m, f1, bc)                                                                          
K_bar, F2= assemble_system(k0, f2,bc)    
_, F3= assemble_system(k0, f3,bc)                                                                            

M_array=M_bar.array()
K_array=K_bar.array()
F1_array= F1.get_local()
F2_array= F2.get_local()
F3_array= F3.get_local()

#modified stiffness matrices
k11= dot(pow(An_1,2)*grad(u),grad(v))*dx 
k22= dot(pow(An_2,2)*grad(u),grad(v))*dx 
k33= dot(pow(An_3,2)*grad(u),grad(v))*dx 
k12= dot(An_1*An_2*grad(u),grad(v))*dx 
k23= dot(An_2*An_3*grad(u),grad(v))*dx 
k13= dot(An_1*An_3*grad(u),grad(v))*dx 
                                                                         
K11,_ = assemble_system(k11, f1, bc)                                                                          
K22,_ = assemble_system(k22, f1, bc) 
K33,_ = assemble_system(k33, f1, bc) 
K12,_ = assemble_system(k12, f1, bc) 
K13,_ = assemble_system(k13, f1, bc)   
K23,_ = assemble_system(k23, f1, bc)                                                                              
K11_bar=3*fractional_matrix_power(M_array,-1/2)@ (K11.array() @fractional_matrix_power(M_array,-1/2))                                     
K22_bar=3*fractional_matrix_power(M_array,-1/2)@ (K22.array() @fractional_matrix_power(M_array,-1/2)) 
K33_bar=3*fractional_matrix_power(M_array,-1/2)@ (K33.array() @fractional_matrix_power(M_array,-1/2)) 
K12_bar=3*fractional_matrix_power(M_array,-1/2)@ (K12.array() @fractional_matrix_power(M_array,-1/2))                                       
K23_bar=3*fractional_matrix_power(M_array,-1/2)@ (K23.array() @fractional_matrix_power(M_array,-1/2)) 
K13_bar=3*fractional_matrix_power(M_array,-1/2)@ (K13.array() @fractional_matrix_power(M_array,-1/2)) 


#modified mass matrices
m11= dot(pow(An_1,2)*u,v)*dx 
m22= dot(pow(An_2,2)*u,v)*dx 
m33= dot(pow(An_3,2)*u,v)*dx 
m12= dot(An_1*An_2*u,v)*dx 
m23= dot(An_2*An_3*u,v)*dx 
m13= dot(An_1*An_3*u,v)*dx 
                                                                         
m11,_ = assemble_system(m11, f1, bc)                                                                          
m22,_ = assemble_system(m22, f1, bc) 
m33,_ = assemble_system(m33, f1, bc) 
m12,_ = assemble_system(m12, f1, bc) 
m13,_ = assemble_system(m13, f1, bc)   
m23,_ = assemble_system(m23, f1, bc)                                                                              
M11_bar=3*fractional_matrix_power(M_array,-1/2)@ (m11.array() @fractional_matrix_power(M_array,-1/2))                                     
M22_bar=3*fractional_matrix_power(M_array,-1/2)@ (m22.array() @fractional_matrix_power(M_array,-1/2)) 
M33_bar=3*fractional_matrix_power(M_array,-1/2)@ (m33.array() @fractional_matrix_power(M_array,-1/2)) 
M12_bar=3*fractional_matrix_power(M_array,-1/2)@ (m12.array() @fractional_matrix_power(M_array,-1/2))                                       
M23_bar=3*fractional_matrix_power(M_array,-1/2)@ (m23.array() @fractional_matrix_power(M_array,-1/2)) 
M13_bar=3*fractional_matrix_power(M_array,-1/2)@ (m13.array() @fractional_matrix_power(M_array,-1/2)) 


u_space= Expression("sin(pi*x[0])", degree=10)
u_e = interpolate(u_space,V).vector().get_local()

d=np.size(u_e)
print('The dimension of the vector is', d)


# Construct the relevant matrices for the resulting ODE systems
Z0=fractional_matrix_power(M_array,1/2) @u_e
Z1=fractional_matrix_power(M_array,1/2) @u_e
L= np.identity(d)
K0= fractional_matrix_power(M_array,-1/2)@ (K_array @fractional_matrix_power(M_array,-1/2))

## Exact solutions 
Z_e = lambda  t: fractional_matrix_power(M_array,1/2)@u_e*np.exp(t)
Zprime_e = lambda  t: fractional_matrix_power(M_array,1/2)@u_e*np.exp(t)


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


#Iterations DG2
t=0;
expt1= lambda t : np.exp(t)*2/k;
expt1_3= lambda t : pow(np.exp(t),3)*2/k;
for it in range(1,nt):
    print(it)
    t+=k
    u_D.t=t
    u_Dprime.t=t
    def phi(t,i):
        if i==0:
            return 1;
        if i==1:
            return 2*(t-time[it-1])/k -1;
        if i==2:
            return 6*(t-time[it-1])**2/(k**2)-6*(t-time[it-1])/k +1;
    def phi_t(t,i):
        if i==0:
            return 0;
        if i==1:
            return 2/k;
        if i==2:
            return 12*(t-time[it-1])/(k**2)-6/k;

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

    M2_bar=[[[[quad(lambda t: int_t(t,i,j,m,n), time[it-1], time[it])[0] for j in range (0,3)] for i in range(0,3)] for n in range(0,3) ] for m in range(0,3)]
    M3_bar=[[[[quad(lambda t: int_t(t,i,j,m,n), time[it-1], time[it])[0] for j in range (0,3)] for i in range(0,3)] for n in range(0,3) ] for m in range(0,3)]
    M5_bar=[[2*(phi(time[it-1],i)*phi(time[it-1],j))*M5*(1-delta(i,j))+ (phi(time[it-1],i)**2)*delta(i,j)*M5 for j in range(0,3)] for i in range(0,3)]
    
    Gexpt1=((np.pi**2)+2)*(fractional_matrix_power(M_array,-1/2)@ F1_array)
    Gexpt3=(3*(np.pi**2)+3)*(fractional_matrix_power(M_array,-1/2)@ F2_array)-(6*np.pi**2)*(fractional_matrix_power(M_array,-1/2)@ F3_array)
    expt2= lambda t : np.exp(t)*phi_t(t,2);
    expt2_3=lambda t : pow(np.exp(t),3)*phi_t(t,2);
    
    #Compute the block diagonal matrix A1
    A_1=M   
    for i in range(1,d):
        A_1= block_diag(A_1,M)    
    #Compute the block matrix B
    B=np.block([[L[i, j]*M2+K0[i,j]*(M3+M5)+K11_bar[i,j]*(M3_bar[0][0]+M5_bar[0][0])+K22_bar[i,j]*(M3_bar[1][1]+M5_bar[1][1])\
    +K33_bar[i,j]*(M3_bar[2][2]+M5_bar[2][2])+K12_bar[i,j]*(M3_bar[0][1]+M5_bar[0][1])+K23_bar[i,j]*(M3_bar[1][2]+M5_bar[1][2])\
    +K13_bar[i,j]*(M3_bar[0][2]+M5_bar[0][2])+np.multiply(M2_bar[0][0],M11_bar[i,j])+np.multiply(M2_bar[1][1],M22_bar[i,j])\
    +np.multiply(M2_bar[2][2],M33_bar[i,j]) + np.multiply(M2_bar[0][1],M12_bar[i,j])+np.multiply(M2_bar[1][2],M23_bar[i,j])\
    +np.multiply(M2_bar[0][2],M13_bar[i,j])for j in range(0,d)] for i in range(0,d)])
    A=A_1+B;
    b[0::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])
    b[1::Dt]=-(K0@Z_minus[it-1,:])-(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])-(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])-(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
    -(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])-(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])-(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(expt1, time[it-1], time[it])[0]*Gexpt1+quad(expt1_3, time[it-1], time[it])[0]*Gexpt3
    b[2::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
    +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(expt2, time[it-1], time[it])[0]*Gexpt1+quad(expt2_3, time[it-1], time[it])[0]*Gexpt3   
    eps =1.0
    tol=1e-10
    maxiter= 30
    iter =0
    while eps>tol and iter<maxiter:
        iter +=1   
        alpha=np.linalg.solve(A,b)

        U_DG_1= fractional_matrix_power(M_array,-1/2)@ alpha[0::Dt]
        U_DG_2= fractional_matrix_power(M_array,-1/2)@ alpha[1::Dt]
        U_DG_3= fractional_matrix_power(M_array,-1/2)@ alpha[2::Dt]

        u_DG_1= Function(V)
        u_DG_1.vector().set_local(U_DG_1)
        
        u_DG_2= Function(V)
        u_DG_2.vector().set_local(U_DG_2)
       
        u_DG_3= Function(V)
        u_DG_3.vector().set_local(U_DG_3)
       
        diff1= u_DG_1.vector()-An_1.vector()
        diff2= u_DG_2.vector()-An_2.vector()
        diff3= u_DG_3.vector()-An_3.vector()
        eps= LA.norm(diff1, ord=np.Inf)+ LA.norm(diff2, ord=np.Inf)+ LA.norm(diff3, ord=np.Inf)
        print('iter= % d: norm =%g' %(iter, eps))

        An_1.assign(u_DG_1)
        An_2.assign(u_DG_2)
        An_3.assign(u_DG_3)

        # stiffness forms
        k11= dot(pow(An_1,2)*grad(u),grad(v))*dx 
        k22= dot(pow(An_2,2)*grad(u),grad(v))*dx 
        k33= dot(pow(An_3,2)*grad(u),grad(v))*dx 
        k12= dot(An_1*An_2*grad(u),grad(v))*dx
        k23= dot(An_2*An_3*grad(u),grad(v))*dx 
        k13= dot(An_1*An_3*grad(u),grad(v))*dx 
        K11,_ = assemble_system(k11, f1, bc)                                                                          
        K22,_ = assemble_system(k22, f1, bc) 
        K33,_ = assemble_system(k33, f1, bc) 
        K12,_ = assemble_system(k12, f1, bc) 
        K13,_ = assemble_system(k13, f1, bc)   
        K23,_ = assemble_system(k23, f1, bc)                                                                              
        K11_bar=3*fractional_matrix_power(M_array,-1/2)@ (K11.array() @fractional_matrix_power(M_array,-1/2))                                     
        K22_bar=3*fractional_matrix_power(M_array,-1/2)@ (K22.array() @fractional_matrix_power(M_array,-1/2)) 
        K33_bar=3*fractional_matrix_power(M_array,-1/2)@ (K33.array() @fractional_matrix_power(M_array,-1/2)) 
        K12_bar=3*fractional_matrix_power(M_array,-1/2)@ (K12.array() @fractional_matrix_power(M_array,-1/2))                                       
        K23_bar=3*fractional_matrix_power(M_array,-1/2)@ (K23.array() @fractional_matrix_power(M_array,-1/2)) 
        K13_bar=3*fractional_matrix_power(M_array,-1/2)@ (K13.array() @fractional_matrix_power(M_array,-1/2)) 

        #modified mass matrices
        m11= dot(pow(An_1,2)*u,v)*dx 
        m22= dot(pow(An_2,2)*u,v)*dx 
        m33= dot(pow(An_3,2)*u,v)*dx 
        m12= dot(An_1*An_2*u,v)*dx 
        m23= dot(An_2*An_3*u,v)*dx 
        m13= dot(An_1*An_3*u,v)*dx 
                                                                         
        M11,_ = assemble_system(m11, f1, bc)                                                   
        M22,_ = assemble_system(m22, f1, bc) 
        M33,_ = assemble_system(m33, f1, bc) 
        M12,_ = assemble_system(m12, f1, bc) 
        M13,_ = assemble_system(m13, f1, bc)   
        M23,_ = assemble_system(m23, f1, bc)                                                                              
        M11_bar=3*fractional_matrix_power(M_array,-1/2)@ (M11.array() @fractional_matrix_power(M_array,-1/2))                                     
        M22_bar=3*fractional_matrix_power(M_array,-1/2)@ (M22.array() @fractional_matrix_power(M_array,-1/2)) 
        M33_bar=3*fractional_matrix_power(M_array,-1/2)@ (M33.array() @fractional_matrix_power(M_array,-1/2)) 
        M12_bar=3*fractional_matrix_power(M_array,-1/2)@ (M12.array() @fractional_matrix_power(M_array,-1/2))                                       
        M23_bar=3*fractional_matrix_power(M_array,-1/2)@ (M23.array() @fractional_matrix_power(M_array,-1/2)) 
        M13_bar=3*fractional_matrix_power(M_array,-1/2)@ (M13.array() @fractional_matrix_power(M_array,-1/2)) 

        #Compute the block diagonal matrix A1
        A_1=M   
        for i in range(1,d):
            A_1= block_diag(A_1,M)    
        #Compute the block matrix B
        B=np.block([[L[i, j]*M2+K0[i,j]*(M3+M5)+K11_bar[i,j]*(M3_bar[0][0]+M5_bar[0][0])+K22_bar[i,j]*(M3_bar[1][1]+M5_bar[1][1])\
        +K33_bar[i,j]*(M3_bar[2][2]+M5_bar[2][2])+K12_bar[i,j]*(M3_bar[0][1]+M5_bar[0][1])+K23_bar[i,j]*(M3_bar[1][2]+M5_bar[1][2])\
        +K13_bar[i,j]*(M3_bar[0][2]+M5_bar[0][2]) + np.multiply(M2_bar[0][0],M11_bar[i,j])+np.multiply(M2_bar[1][1],M22_bar[i,j])\
        +np.multiply(M2_bar[2][2],M33_bar[i,j]) + np.multiply(M2_bar[0][1],M12_bar[i,j])+np.multiply(M2_bar[1][2],M23_bar[i,j])\
        +np.multiply(M2_bar[0][2],M13_bar[i,j])for j in range(0,d)] for i in range(0,d)])
        A=A_1+B;
        b[0::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])
        b[1::Dt]=-(K0@Z_minus[it-1,:])-(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])-(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])-(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
        -(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])-(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])-(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])+Zprime_minus[it-1,:]*2/k+quad(expt1, time[it-1], time[it])[0]*Gexpt1+quad(expt1_3, time[it-1], time[it])[0]*Gexpt3
        b[2::Dt]= (K0@Z_minus[it-1,:])+(phi(time[it-1],0)**2)*(K11_bar@Z_minus[it-1,:])+(phi(time[it-1],1)**2)*(K22_bar@Z_minus[it-1,:])+(phi(time[it-1],2)**2)*(K33_bar@Z_minus[it-1,:])\
        +(2*phi(time[it-1],0)*phi(time[it-1],1))*(K12_bar@Z_minus[it-1,:])+(2*phi(time[it-1],1)*phi(time[it-1],2))*(K23_bar@Z_minus[it-1,:])+(2*phi(time[it-1],2)*phi(time[it-1],0))*(K13_bar@Z_minus[it-1,:])+Zprime_minus[it-1,:]*(-6/k)+quad(expt2, time[it-1], time[it])[0]*Gexpt1+quad(expt2_3, time[it-1], time[it])[0]*Gexpt3
 
    convergence = 'convergence after %d Picard iterations' % iter
    if iter>= maxiter:
        convergence = ' no' + convergence
    print ('''
        Solution of the nonlinear wave problem u_tt+a'(u)u_t-u_txx-(b(u))_xx =f  with a(u)=u+u^3, b(u)=u+u^3,
        f=(2pi^2+2)*exp(t)*sin(pi*x)+(3pi^2+3)*exp(3*t)*sin^3(pi*x)-6pi^2*exp(3*t)*cos^2(pi*x)*sin(pi*x),
        u(0,x)=sin(pi*x), u_t(0,x)= sin(pi*x) and u=0 at x=0 and  x=1.
        %s
        ''' % ( convergence)) 
    Z_minus[it,:]=  alpha[0::Dt]+alpha[1::Dt]+alpha[2::Dt]
    Z_plus[it-1,:]= alpha[0::Dt]-alpha[1::Dt]+alpha[2::Dt]
    Zprime_minus[it,:]= alpha[1::Dt]*2/k+alpha[2::Dt]*6/k
    Zprime_plus[it-1,:]= alpha[1::Dt]*2/k-alpha[2::Dt]*6/k
# Compute the error in L2 norm
U_end_DG= fractional_matrix_power(M_array,-1/2)@ Z_minus[nt-1,:]
Uprime_end_DG= fractional_matrix_power(M_array,-1/2)@ Zprime_minus[nt-1,:]


u_end_DG= Function(V)
u_end_DG.vector().set_local(U_end_DG)

uprime_end_DG= Function(V)
uprime_end_DG.vector().set_local(Uprime_end_DG)

error1=errornorm(u_D, u_end_DG, 'H1')
error2=errornorm(u_Dprime, uprime_end_DG, 'L2')
error=error1+error2
print('The error ||u-u_DG||_1+||dot u- dot u_DG|| is', error )
print('The L2 error ||dot u- dot u_DG|| is', error2 )



# T=1, tol=1e-10  CG2 in space k=h^2--> expected order O(k)
# |  r   | h         |  k= h^2  |  L^2 norm   |   rate   |  
# |------|:----------|:--------:|:-----------:|:--------:|
# |  2   |5.0000e-1  |2.5000e-1 | 6.2306e-1   |   ----   |
# |      |2.5000e-1  |6.2500e-2 | 1.5307e-1   |  1.0126  | 
# |      |2.0000e-1  |4.0000e-2 | 9.0682e-2   |  1.1731  |  
# |      |1.2500e-1  |1.5625e-2 | 3.5950e-2   |  0.9843  |   
# |      |1.0000e-1  |1.0000e-2 | 2.2515e-2   |  1.0486  | 

# T=1, tol=1e-10  CG3 in space k=h^2--> expected order O(k^{3/2})
# |  r   | h         |  k= h^2  |  L^2 norm   |   rate   |  
# |------|:----------|:--------:|:-----------:|:--------:|
# |  3   |5.0000e-1  |2.5000e-1 | 9.0111e-2   |   ----   |
# |      |2.5000e-1  |6.2500e-2 | 9.6758e-3   |  1.6096  | 
# |      |2.0000e-1  |4.0000e-2 | 4.7781e-3   |  1.5810  |  
# |      |1.2500e-1  |1.5625e-2 | 1.1819e-3   |  1.4861  |   
# |      |1.0000e-1  |1.0000e-2 | 5.9763e-4   |  1.5280  | 

# T=1, tol=1e-10 CG4 in space k=h^2--> expected order O(k^2)
# |  r   | h         |  k= h^2  |  L^2 norm   |   rate   |  
# |------|:----------|:--------:|:-----------:|:--------:|
# |  4   |5.0000e-1  |2.5000e-1 | 1.2923e-2   |   ----   |
# |      |2.5000e-1  |6.2500e-2 | 5.3962e-4   |  2.2909  | 
# |      |2.0000e-1  |4.0000e-2 | 1.9952e-4   |  2.2294  |  
# |      |1.2500e-1  |1.5625e-2 | 3.0814e-5   |  1.9872  |   
# |      |1.0000e-1  |1.0000e-2 | 1.1982e-5   |  2.1165  | 
clf()
# Computed errors ||dot u- dot u_DG||_1+||u-u_DG|| versus 1/k (loglog scale) and r=2,3,4
Delta_t= np.array([ 1/2.5000e-1, 1/6.2500e-2, 1/4.0000e-2, 1/1.5625e-2, 1/1.0000e-2]);
error_2= np.array([ 6.2306e-1,  1.5307e-1, 9.0682e-2, 3.5950e-2, 2.2515e-2]);
error_3= np.array([ 9.0111e-2,  9.6758e-3, 4.7781e-3, 1.1819e-3, 5.9763e-4]);
error_4= np.array([ 1.2923e-2,  5.3962e-4, 1.9952e-4, 3.0814e-5, 1.1982e-5]);
loglog(Delta_t, error_2, label='CG 2', linestyle='-', marker='o', color='blue', linewidth=3)
loglog(Delta_t, error_3, label='CG 3', linestyle=':', marker='p', color='red', linewidth=3)
loglog(Delta_t, error_4, label='CG 4', linestyle='--',marker='D', color='black', linewidth=3)
title('Plot of $||u- u_\mathrm{DG}||_{H^1}+||\dot{u}- \dot{u}_\mathrm{DG}||_{L^2}$ versus 1/k for DG2 in time')
legend(loc = 'best')
grid()
savefig('1D_nonlinear_wave2_DG2_L2error.png', dpi=300, bbox_inches='tight')

    




