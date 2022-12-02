# Discontinuous-in-time Finite Element Method

The discontinuous Galerkin (DG) time-stepping method is implemented for various second-order hyperbolic partial differential equations (PDEs). 
The proposed numerical scheme combines the hp-version discontinuous Galerkin finite element method (hp-DGFEM) in the time direction with an $H^1(\Omega)$-conforming finite element approximation for the spatial variables.

## 1. Linear hyperbolic equations of 2nd order 
### a) Implementation of the scheme for a scalar linear wave equation. 
    a.1 1D linear wave equation using DG-q (with q=2,3,4,5) for time discretization
    a.2 1D linear wave equation using alpha method for time discretization 
   
### b) Implementation of the scheme for a 2D linearised elastodynamics equation.    
    b.1 2D linearized elastodynamics equation using DG-q (with q=2,3,4) for time discretization
    b.2 1D linearized elastodynamcis equation using alpha method for time discretization 
   
     
 
   
## 2. Nonlinear elastodynamics equations 
### a) Implementation of the scheme for a scalar nonlinear elastodynamics problem.
    a.1 1D non-linear elastoynamics equation using DG-2 elements for time discretization
    a.2 1D non-linear elastoynamics equation using DG-3 elements for time discretization
    a.3 1D non-linear elastoynamics equation using DG-4 elements for time discretization

## 3. Nonlinear damped wave equations.
### a) Implementation of the scheme for a scalar nonlinear damped wave equation.
     a.1 1D non-linear damped wave equation using DG-2 elements for time discretization
     a.2 1D non-linear damped wave equation using DG-3 elements for time discretization
     a.3 1D non-linear damped wave equation using DG-4 elements for time discretization
### b) Implementation of the scheme for a scalar nonlinear wave equation (no extra stability term).
    b.1 1D non-linear wave equation using DG-2 elements for time discretization
    b.2 1D non-linear wave equation using DG-3 elements for time discretization
    b.3 1D non-linear wave equation using DG-4 elements for time discretization



   
     
     
