# La idea de este script es correr lo mismo que el jupyter pero con terminal, evitando incompatibilidades de kernel de jupyter con librerias



from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import basix.ufl

# Malla y Espacio de Funciones (2D, memoria) 
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1) 
gdim = domain.geometry.dim

cell_name_str = domain.topology.cell_name() 
P2_vec = basix.ufl.element("Lagrange", cell_name_str, 2, shape=(gdim,))
P1_scal = basix.ufl.element("Lagrange", cell_name_str, 1)

W_elem = basix.ufl.mixed_element([P2_vec, P1_scal, P1_scal, P1_scal])
W = fem.functionspace(domain, W_elem)

# Variables y Funciones 
wh = fem.Function(W)
wh_o = fem.Function(W) 

(d_star, phi0_star, lam_star, mu_star) = ufl.TestFunctions(W)

d, phi0, lam, mu = ufl.split(wh)
d_o, phi0_o, lam_o, mu_o = ufl.split(wh_o)

# Parámetros (Basados paper)
dt_val = 0.01                 # Paso de tiempo (dt) 
t_ramp = 0.1                  # Tiempo de rampa de carga 
t_final = 2.0                 # Tiempo final
num_steps = int(t_final / dt_val)
dt = fem.Constant(domain, dt_val)

# Poroelasticidad
k_val = 2.0e-7                
rho_f = fem.Constant(domain, 1.0) 
phi_bar = fem.Constant(domain, 0.1) 

# Carga (Fuente)
pa = fem.Constant(domain, 10000.0)   
beta_max = 1.0e-4                     
beta = fem.Constant(domain, 0.0)      

# Cinematica y tensores

I = ufl.Identity(gdim)
f = I + ufl.grad(d) #
j = ufl.det(f)      
F = ufl.inv(f)      
J = ufl.det(F)     

# Modelo material hipereslastico

C = fem.Constant(domain, 880.0) 
B = fem.Constant(domain, 5.0e4)

# Anisotropia parametros inspo script paper

bff = 20.0
bss = 10.0
bnn = 20.0
bfs = 10.0
bfn = 10.0
bsn = 10.0

# Direcciones de fibra (alineadas con los ejes para el cubo) 
f0 = ufl.as_vector([1.0, 0.0, 0.0])
s0 = ufl.as_vector([0.0, 1.0, 0.0])
n0 = ufl.as_vector([0.0, 0.0, 1.0])

C_tensor = F.T * F
E_green = 0.5 * (C_tensor - I) # Tensor de Green-Lagrange

E_ff = ufl.inner(E_green * f0, f0)
E_ss = ufl.inner(E_green * s0, s0)
E_nn = ufl.inner(E_green * n0, n0)
E_fs = ufl.inner(E_green * f0, s0)
E_fn = ufl.inner(E_green * f0, n0)
E_sn = ufl.inner(E_green * s0, n0)

# Exponente Q 
Q = bff * E_ff**2 + bss * E_ss**2 + bnn * E_nn**2 + \
    bfs * (E_fs**2) + bfn * (E_fn**2) + bsn * (E_sn**2)

# Anisotropica
S_ff = 2.0 * bff * E_ff
S_ss = 2.0 * bss * E_ss
S_nn = 2.0 * bnn * E_nn
S_fs = 2.0 * bfs * E_fs
S_fn = 2.0 * bfn * E_fn
S_sn = 2.0 * bsn * E_sn

S = S_ff * ufl.outer(f0, f0) + S_ss * ufl.outer(s0, s0) + S_nn * ufl.outer(n0, n0) + \
    S_fs * (ufl.outer(f0, s0) + ufl.outer(s0, f0)) + \
    S_fn * (ufl.outer(f0, n0) + ufl.outer(n0, f0)) + \
    S_sn * (ufl.outer(s0, n0) + ufl.outer(n0, s0))

P_aniso = C * ufl.exp(Q) * (F * S)

# Volumetrica

P_vol = B * J**2 * ufl.inv(F.T)

P = P_aniso + P_vol

# Parametros presion paper script basado

q1 = fem.Constant(domain, 1.333)
q2 = fem.Constant(domain, 550.0)
q3 = fem.Constant(domain, 10.0)

def dp_P(Phi):
    # Agregamos 'eps' para estabilidad numérica, evitando ln(0)
    eps = fem.Constant(domain, 1e-10)
    return (q1/q3) * ufl.exp(q3 * Phi) + q2 * ufl.ln(q3 * Phi + eps)

Phi_actual = J * phi_bar # Phi = J * phi_bar
Phi_ref = phi0

p_tilde = dp_P(Phi_actual) - dp_P(ufl.conditional(ufl.gt(Phi_ref, 0.0), Phi_ref, 1e-10))

theta = -beta * (mu - pa) # mu es la variable de presión mixta 


phi0_dt = (phi0 - phi0_o) / dt

# Equilibrio Mecánico 
F_mech = (ufl.inner(j * P * ufl.inv(f.T), ufl.grad(d_star)) + ufl.inner(lam, ufl.div(d_star))) * ufl.dx

# Incompresibilidad
F_incomp = ufl.inner(j * (1 - phi0) - (1 - phi_bar), lam_star) * ufl.dx

# Ecuación de Flujo (con k_val y theta)
F_poro_A = (-ufl.inner(phi0_dt, mu_star) \
            + ufl.inner(k_val * ufl.grad(mu), ufl.grad(mu_star)) \
            - ufl.inner(1/rho_f * theta, mu_star)) * ufl.dx

# Definición de Presión
F_poro_B = ufl.inner(mu - p_tilde, phi0_star) * ufl.dx

F_total = F_mech + F_incomp + F_poro_A + F_poro_B

# Condiciones Nulas testeo rapido

def left_boundary(x):
    return np.isclose(x[0], 0)
left_facets = mesh.locate_entities_boundary(domain, gdim - 1, left_boundary)
left_dofs_x = fem.locate_dofs_topological(W.sub(0).sub(0), gdim - 1, left_facets)
bc_left_x = fem.dirichletbc(0.0, left_dofs_x, W.sub(0).sub(0))

def bottom_boundary(x):
    return np.isclose(x[1], 0)
bottom_facets = mesh.locate_entities_boundary(domain, gdim - 1, bottom_boundary)
bottom_dofs_y = fem.locate_dofs_topological(W.sub(0).sub(1), gdim - 1, bottom_facets)
bc_bottom_y = fem.dirichletbc(0.0, bottom_dofs_y, W.sub(0).sub(1))

def front_boundary(x):
    return np.isclose(x[2], 0)

front_facets = mesh.locate_entities_boundary(domain, gdim - 1, front_boundary)
front_dofs_z = fem.locate_dofs_topological(W.sub(0).sub(2), gdim - 1, front_facets)
bc_front_z = fem.dirichletbc(0.0, front_dofs_z, W.sub(0).sub(2))

bcs = [bc_left_x, bc_bottom_y, bc_front_z] 

# Solver no lineal
problem = NonlinearProblem(F_total, wh, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6 
solver.max_it = 50 

log.set_log_level(log.LogLevel.INFO)

phi0_init_expr = fem.Expression(phi_bar, W.sub(1).element.interpolation_points())
wh_o.sub(1).interpolate(phi0_init_expr)
wh.sub(1).interpolate(phi0_init_expr)

print(f"Iniciando simulación IPP (Test 2 - Cubo 3D) [fuente: 432]...")
print(f"Parámetros: dt={dt_val}, t_ramp={t_ramp}, k={k_val}, pa={pa.value}")

V_out_P1_vec = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
d_out = fem.Function(V_out_P1_vec)
d_out.name = "Desplazamiento_Inverso_P1"

phi0_sol = wh.sub(1)
mu_sol = wh.sub(3)
phi0_sol.name = "Porosidad_Referencia"
mu_sol.name = "Presion_Mixta"

xdmf = io.XDMFFile(MPI.COMM_WORLD, "IPP_cubo_resultado.xdmf", "w")
xdmf.write_mesh(domain)

d_hat_sol_P2 = wh.sub(0) 
d_out.interpolate(d_hat_sol_P2) 

xdmf.write_function(d_out, 0.0) 
xdmf.write_function(phi0_sol, 0.0)
xdmf.write_function(mu_sol, 0.0)

norm_phi0_form = fem.form(ufl.inner(phi0_sol, phi0_sol) * ufl.dx)
norm_phi0_o_form = fem.form(ufl.inner(wh_o.sub(1), wh_o.sub(1)) * ufl.dx)

for i in range(num_steps):
    t_current = (i + 1) * dt_val
    
    ramp_factor = min(1.0, t_current / t_ramp)
    beta.value = beta_max * ramp_factor
    
    print(f"Paso {i+1}/{num_steps}, (Pseudo) Tiempo {t_current:.3f}, Carga (beta): {ramp_factor*100:.1f}%")

    try:
        n_iter, converged = solver.solve(wh)
        if not converged:
            print("Newton no convergió.")
            break
    except Exception as e:
        print(f"Fallo el solver: {e}")
        break

    wh_o.x.array[:] = wh.x.array
    
    if (i+1) % 10 == 0:
        d_out.interpolate(wh.sub(0)) 
        xdmf.write_function(d_out, t_current) 
        xdmf.write_function(phi0_sol, t_current)
        xdmf.write_function(mu_sol, t_current)
    
    norm_val = np.sqrt(fem.assemble_scalar(norm_phi0_form))
    norm_val_o = np.sqrt(fem.assemble_scalar(norm_phi0_o_form))
    norm_phi0_change = norm_val - norm_val_o
    
    if np.abs(norm_phi0_change) < 1e-7 and ramp_factor == 1.0:
        print(f"Convergencia de estado estacionario alcanzada en el paso {i+1}.")
        break

xdmf.close()
print("Simulación terminada.")

norm_val_final = np.sqrt(fem.assemble_scalar(norm_phi0_form))
print(f"Norma L2 de la porosidad de referencia (phi0) final: {norm_val_final:.4e}")