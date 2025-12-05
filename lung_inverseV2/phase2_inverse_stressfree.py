"""
FASE 2: Problema Inverso - Recuperar Configuración Libre de Tensiones
======================================================================

Dado:
- Malla en configuración deformada (con prestress)
- Campos observados: u_obs, Γ_obs
- Presión alveolar P_alv

Encontrar:
- Mapeo inverso w: Ω_deformed → Ω_stress-free
- O equivalentemente: configuración de referencia sin tensiones

Enfoque:
- Formulación de deformación inversa (Govindjee & Mihalic, 1996)
- La "configuración actual" es la observada (CT)
- Buscamos la configuración de referencia donde S = 0 sin cargas

Formulación:
- En configuración observada: F_obs = I + ∇u_true
- Mapeo inverso: φ⁻¹: Ω_current → Ω_reference
- f = ∇φ⁻¹ (gradiente del mapeo inverso)
- F_forward = f⁻¹
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
from firedrake.adjoint import *
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# ============================================================
# 1. PARÁMETROS (mismos que Fase 1)
# ============================================================

# Geometría alveolar
Phi_R = 0.74
R_RI = 0.11

# Material Fung
c_fung = 2.5e-3
a_fung = 0.433
b_fung = -0.61

# Surfactante
gamma_0 = 70.0e-6
gamma_inf = 22.2e-6
gamma_min = 0.0
Gamma_inf = 3.0e-9
m1 = 47.8e-6
m2 = 140.0e-6
Gamma_max = Gamma_inf * (1.0 + (gamma_inf - gamma_min) / m2)

print("="*60)
print("FASE 2: PROBLEMA INVERSO")
print("="*60)

# ============================================================
# 2. CARGAR DATOS DE FASE 1
# ============================================================

input_dir = "phase1_results"
data = np.load(os.path.join(input_dir, "forward_data.npz"))

u_obs_data = data['u_data']
Gamma_obs_data = data['Gamma_data']
P_alv_value = data['P_alv_final']

print(f"\nDatos cargados de Fase 1:")
print(f"  P_alv = {P_alv_value*1000:.2f} Pa")
print(f"  |u_obs| = {np.linalg.norm(u_obs_data):.4e}")

# ============================================================
# 3. MALLA Y ESPACIOS
# ============================================================

Lx, Ly = 10.0, 10.0
nx, ny = 20, 20

mesh = fd.RectangleMesh(nx, ny, Lx, Ly)

V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "DG", 0)
W = fd.FunctionSpace(mesh, "CG", 1)  # Para el campo de predeformación

dim = mesh.geometric_dimension()
I = fd.Identity(dim)

# ============================================================
# 4. CARGAR OBSERVACIONES
# ============================================================

u_obs = fd.Function(V, name="u_observed")
u_obs.dat.data[:] = u_obs_data

Gamma_obs = fd.Function(Q, name="Gamma_observed")
Gamma_obs.dat.data[:] = Gamma_obs_data

P_alv = fd.Constant(P_alv_value)

print(f"\nObservaciones cargadas:")
print(f"  ||u_obs|| = {fd.norm(u_obs):.4e}")
print(f"  Γ_obs: [{Gamma_obs.dat.data_ro.min():.2e}, {Gamma_obs.dat.data_ro.max():.2e}]")

# ============================================================
# 5. FORMULACIÓN DEL PROBLEMA INVERSO
# ============================================================
"""
Enfoque: Predeformación multiplicativa

La idea es que la configuración observada (CT) tiene un prestress.
Modelamos esto como:
- F_total = F_elastic · F_prestress
- Donde F_prestress representa la "deformación previa" que genera el prestress

Para el problema inverso:
- Conocemos el estado deformado
- Queremos encontrar F_prestress tal que en la config. de referencia S = 0

Parametrizamos F_prestress con un campo escalar θ (isótropo):
- F_g = (1 + θ) · I
- θ > 0: expansión (la referencia es más pequeña)
- θ < 0: contracción (la referencia es más grande)

Pero como vimos antes, θ isotrópico se cancela en F_bar.
Por eso usamos un enfoque diferente: campo de predeformación w.
"""

# Campo de predeformación (lo que buscamos)
w = fd.Function(V, name="predeformation")

# Variables de estado
u = fd.Function(V, name="displacement")
v_test = fd.TestFunction(V)

# ============================================================
# 6. MODELO CON PREDEFORMACIÓN
# ============================================================

def compute_model(u_field, w_field, Gamma_field, P_alv_const):
    """
    Modelo forward con predeformación.
    
    El gradiente de deformación efectivo es:
    F = I + ∇(u + w)
    
    donde:
    - u es el desplazamiento elástico
    - w es la predeformación (incógnita del problema inverso)
    
    En la configuración de referencia (sin cargas), queremos que
    el material esté libre de tensiones cuando u = 0.
    """
    # Gradiente de deformación total
    F = I + fd.grad(u_field + w_field)
    J = fd.det(F)
    C = F.T * F
    invC = fd.inv(C)
    
    # Tensor de Green-Lagrange
    E = fd.variable(0.5 * (C - I))
    
    # Invariantes
    J1 = fd.tr(E)
    J2 = 0.5 * (fd.tr(E)**2 - fd.tr(E * E))
    
    # Energía Fung
    Psi_el = c_fung * (fd.exp(a_fung * J1**2 + b_fung * J2) - 1.0)
    S_el = fd.diff(Psi_el, E)
    
    # Porosidad
    Phi = fd.max_value(J - 1.0 + Phi_R, 1e-4)
    
    # Tensión superficial
    ratio = Gamma_field / Gamma_inf
    gamma_lang = gamma_0 - m1 * ratio
    gamma_insol = gamma_inf - m2 * (ratio - 1.0)
    gamma = fd.conditional(fd.lt(Gamma_field, Gamma_inf), gamma_lang, gamma_insol)
    gamma = fd.max_value(gamma, gamma_min)
    
    # Presión de colapso
    P_gamma = (2.0 * gamma / R_RI) * ((Phi_R / Phi)**(1.0/3.0))
    
    # Tensor total
    S_total = S_el + (P_gamma - P_alv_const) * J * invC
    
    return F, S_total, J, gamma, P_gamma

# ============================================================
# 7. FUNCIONAL DEL PROBLEMA INVERSO
# ============================================================
"""
Queremos encontrar w tal que:

1. El desplazamiento predicho u(w) + w ≈ u_obs
   (la posición total debe coincidir con la observada)

2. En la configuración de referencia (con w, sin cargas P=0),
   el material debería estar libre de tensiones.

Funcional:
J(w) = J_geom + β·J_reg

donde:
- J_geom = ½∫|u(w) + w - u_obs|² dx
- J_reg = ½∫|∇w|² dx (regularización para suavidad)
"""

# Parámetros de regularización
gamma_reg = fd.Constant(1e-4)
beta_stress = fd.Constant(1e-2)  # Peso del término de stress-free

# ============================================================
# 8. RESOLUCIÓN DEL PROBLEMA FORWARD
# ============================================================

def solve_forward(w_field, Gamma_field, P_alv_const):
    """Resolver problema forward dado w"""
    
    # Limpiar tape
    tape = get_working_tape()
    tape.clear_tape()
    
    # Crear nueva función para u
    u_sol = fd.Function(V)
    v = fd.TestFunction(V)
    
    # Modelo
    F, S_total, J, gamma, P_gamma = compute_model(u_sol, w_field, Gamma_field, P_alv_const)
    
    # Residuo
    F_stress = F * S_total
    Residual = fd.inner(F_stress, fd.grad(v)) * fd.dx
    
    # BCs
    bcs = [
        fd.DirichletBC(V, fd.Constant((0.0, 0.0)), 1),
        fd.DirichletBC(V.sub(1), 0.0, 3),
    ]
    
    # Resolver
    problem = fd.NonlinearVariationalProblem(Residual, u_sol, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters={
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 50,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    })
    
    try:
        solver.solve()
        return u_sol, True
    except fd.ConvergenceError:
        return u_sol, False

# ============================================================
# 9. DIAGNÓSTICO DE IDENTIFICABILIDAD
# ============================================================

print("\n" + "="*60)
print("DIAGNÓSTICO DE IDENTIFICABILIDAD")
print("="*60)

with stop_annotating():
    # Test 1: w = 0
    w.assign(fd.Constant((0.0, 0.0)))
    u_w0, success = solve_forward(w, Gamma_obs, P_alv)
    
    if success:
        # Misfit con w = 0
        diff_w0 = fd.assemble(fd.inner(u_w0 + w - u_obs, u_w0 + w - u_obs) * fd.dx)
        print(f"\nCon w = 0:")
        print(f"  ||u(w=0)||  = {fd.norm(u_w0):.6e}")
        print(f"  ||u + w - u_obs||² = {diff_w0:.6e}")
    
    # Test 2: w = u_obs (caso trivial donde u = 0 es solución)
    w.assign(u_obs)
    u_wobs, success = solve_forward(w, Gamma_obs, P_alv)
    
    if success:
        diff_wobs = fd.assemble(fd.inner(u_wobs + w - u_obs, u_wobs + w - u_obs) * fd.dx)
        print(f"\nCon w = u_obs:")
        print(f"  ||u(w=u_obs)|| = {fd.norm(u_wobs):.6e}")
        print(f"  ||u + w - u_obs||² = {diff_wobs:.6e}")

# ============================================================
# 10. OPTIMIZACIÓN
# ============================================================

print("\n" + "="*60)
print("OPTIMIZACIÓN ADJOINT")
print("="*60)

# Inicializar w = 0
w.assign(fd.Constant((0.0, 0.0)))

n_eval = [0]

def objective_and_gradient(w_vec):
    """Evaluar funcional y gradiente"""
    n_eval[0] += 1
    
    # Asignar w
    w.dat.data[:] = w_vec.reshape(w.dat.data.shape)
    
    # Limpiar tape
    tape = get_working_tape()
    tape.clear_tape()
    
    # Resolver forward
    u.assign(fd.Constant((0.0, 0.0)))
    
    F, S_total, J_det, gamma, P_gamma = compute_model(u, w, Gamma_obs, P_alv)
    
    # Residuo
    F_stress = F * S_total
    Residual = fd.inner(F_stress, fd.grad(v_test)) * fd.dx
    
    bcs = [
        fd.DirichletBC(V, fd.Constant((0.0, 0.0)), 1),
        fd.DirichletBC(V.sub(1), 0.0, 3),
    ]
    
    try:
        fd.solve(Residual == 0, u, bcs=bcs,
                 solver_parameters={
                     "snes_type": "newtonls",
                     "snes_max_it": 30,
                     "ksp_type": "preonly",
                     "pc_type": "lu",
                 })
    except:
        return np.inf, np.zeros_like(w_vec)
    
    # Funcional
    J_geom = 0.5 * fd.assemble(fd.inner(u + w - u_obs, u + w - u_obs) * fd.dx)
    J_reg = 0.5 * gamma_reg * fd.assemble(fd.inner(fd.grad(w), fd.grad(w)) * fd.dx)
    J_total = J_geom + J_reg
    
    # Gradiente
    control = Control(w)
    rf = ReducedFunctional(J_total, control)
    
    try:
        dJ = rf.derivative()
        grad_vec = dJ.dat.data_ro.flatten().copy()
    except:
        grad_vec = np.zeros_like(w_vec)
    
    if n_eval[0] <= 5 or n_eval[0] % 10 == 0:
        print(f"  [eval {n_eval[0]}] J = {float(J_total):.4e} "
              f"(geom={float(J_geom):.2e}, reg={float(J_reg):.2e})")
    
    return float(J_total), grad_vec

# Optimización
print("\nIniciando L-BFGS-B...")

w0 = w.dat.data_ro.flatten().copy()

result = scipy_minimize(
    objective_and_gradient,
    w0,
    method="L-BFGS-B",
    jac=True,
    options={
        "maxiter": 100,
        "ftol": 1e-10,
        "gtol": 1e-6,
        "disp": True,
    }
)

print(f"\nResultado:")
print(f"  Mensaje: {result.message}")
print(f"  Iteraciones: {result.nit}")
print(f"  J final: {result.fun:.6e}")

# Asignar resultado
w.dat.data[:] = result.x.reshape(w.dat.data.shape)

# ============================================================
# 11. EVALUACIÓN FINAL
# ============================================================

print("\n" + "="*60)
print("EVALUACIÓN FINAL")
print("="*60)

with stop_annotating():
    # Resolver con w óptimo
    u_final, _ = solve_forward(w, Gamma_obs, P_alv)
    
    # Error de reconstrucción
    error_geom = fd.sqrt(fd.assemble(fd.inner(u_final + w - u_obs, u_final + w - u_obs) * fd.dx))
    u_obs_norm = fd.sqrt(fd.assemble(fd.inner(u_obs, u_obs) * fd.dx))
    
    print(f"\nReconstrucción geométrica:")
    print(f"  ||u + w - u_obs|| = {error_geom:.6e}")
    print(f"  ||u_obs|| = {u_obs_norm:.6e}")
    print(f"  Error relativo: {error_geom/u_obs_norm*100:.2f}%")
    
    print(f"\nPredeformación recuperada:")
    print(f"  ||w|| = {fd.norm(w):.6e}")
    print(f"  w: min = {w.dat.data_ro.min():.4e}, max = {w.dat.data_ro.max():.4e}")

# ============================================================
# 12. GUARDAR RESULTADOS
# ============================================================

output_dir = "phase2_results"
os.makedirs(output_dir, exist_ok=True)

from firedrake.output import VTKFile

outfile = VTKFile(os.path.join(output_dir, "inverse_result.pvd"))
w.rename("predeformation_w")
u_final.rename("displacement_u")

# Campo de error
error_field = fd.Function(V, name="position_error")
error_field.assign(u_final + w - u_obs)

outfile.write(w, u_final, u_obs, error_field)

# Guardar datos
np.savez(os.path.join(output_dir, "inverse_data.npz"),
         w_data=w.dat.data_ro[:],
         u_final_data=u_final.dat.data_ro[:],
         error_geom=float(error_geom))

print(f"\nResultados guardados en: {output_dir}/")

print("\n" + "="*60)
print("FASE 2 COMPLETADA")
print("="*60)
print("\nLa predeformación w define la configuración libre de tensiones.")
print("Posición stress-free = X - w(X)")
