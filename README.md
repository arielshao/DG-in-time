# Discontinuous-in-time Finite Element Methods

A discontinuous Galerkin (DG) time-stepping method is implemented for various second-order hyperbolic partial differential equations (PDEs). 
The proposed numerical scheme combines the hp-version discontinuous Galerkin finite element method (hp-DGFEM) in the time direction with an $H^1(\Omega)$-conforming finite element approximation for the spatial variables.

## 1. Linear hyperbolic equations of 2nd order 
### a) Implementation of the scheme for a scalar linear wave equation. 
      We consider the one-dimensional wave problem
\begin{cases}
\ddot u(x,t)+2\gamma\dot u(x,t)+\gamma^2u(x,t)-\partial_{xx} u(x,t)=f(x,t) &\quad \text{ in } (0,1)\times (0,T],\\
u(0, t) = u(1,t)= 0 &\quad\text{ for all } t\in (0,T], \\
u(x,0)=u_0(x), \quad \dot u(x,0)=u_1(x).
\end{cases}
Here, we set $T=1$, $\gamma=1$ and let $u_0$, $u_1$ and $f$ be chosen such that the exact solution is $u(x,t)=\sin(\sqrt{2}\pi t)\sin(\pi x).$
That is, $u_0(x)\equiv 0$, $u_1(x)= \sqrt{2}\pi \sin(\pi x)$, and $$f(x,t)=[(-\pi^2+\gamma^2)\sin(\sqrt{2}\pi t)+2\sqrt{2}\gamma\pi\cos(\sqrt{2}\pi t)]\sin(\pi x).$$
Here I include the implementations of 
   a.1 1D linear wave equation using DG-q (with q=2,3,4,5) for time discretization
   a.2 1D linear wave equation using alpha method for time discretization 
   
### b) Implementation of the scheme for a 2D linearised elastodynamics equation.    
   
  Now we consider a two-dimensional linearised elastodynamics problem. For $T>0$, find $\mathbf{u}\colon\Omega\times [0, T]\to\mathbb{R}^2$ such that 
\begin{cases}
\rho \ddot{ \mathbf{u}}+2\rho\gamma\dot{\mathbf{u}}+\rho\gamma^2\mathbf{u}-\nabla \cdot \bm{\sigma}=\mathbf{f} &\quad \text{ in } \Omega\times (0,T],\\
\mathbf{u}=0 &\quad\text{ on } \partial\Omega\times (0,T], \\
\mathbf{u}(x,0)=\mathbf{u}_0(x), \quad \dot{\mathbf{u}}(x,0)=\mathbf{u}_1(x) &\quad \text{ in }\Omega.
\end{cases}
Here $\mathbf{f}\in L^2(0,T, L^2(\Omega))$ is the source term, and $\rho\in L^{\infty}(\Omega)$ is such that $\rho=\rho(\mathbf{x})>0$ for almost any $\mathbf{x}\in\Omega$. The stress tensor $\sigma$ is defined through Hooke's law, that is, $\sigma =2\mu \varepsilon +\lambda \mathrm{tr}(\varepsilon)\mathrm{Id},$
where $\mathrm{Id}$ is the identity matrix, $\mathrm{tr}$ is the trace operator, and 
$$\varepsilon(\mathbf{u})=\frac{1}{2}(\nabla \mathbf{u}+\nabla \mathbf{u}^T).$$
Note that we assume $\partial\Omega\in C^2$ in the problem set up (cf. equation \ref{setup}) to apply the duality argument in the convergence analysis (cf. Section \ref{fullydiscrete}). In fact, it is sufficient to assume that $\Omega$ is a convex polygon for $d=2$. We consider $\Omega=(0,1)\times (0,1)$, $T=1$, and set $\rho=1, \gamma=1, \lambda=1$ and $\mu=1$. Here we choose $\mathbf{u}_0$, $\mathbf{u}_1$ and $\mathbf{f}$ such that the exact solution is 
$$\mathbf{u}(x,t)=\sin(\sqrt{2}\pi t)\begin{bmatrix}
-\sin^2(\pi x)\sin(2\pi y)\\
\sin(2\pi x)\sin^2(\pi y)
\end{bmatrix}.$$  
Here I include the implementations of 
   b.1 2D linearised elastodynamics equation using DG-q (with q=2,3,4) for time discretization
   b.2 1D linearised elastodynamcis equation using alpha method for time discretization 
   
     
 
   
## 2. Nonlinear elastodynamics equations 

## 3. Nonlinear damped wave equations.

   
     
     
