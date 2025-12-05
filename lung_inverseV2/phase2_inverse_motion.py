"""
FASE 2 (REVISADA): Recuperación de Configuración Stress-Free
=============================================================

PROBLEMA CON FORMULACIÓN ADITIVA:
---------------------------------
Como discutimos, F = I + ∇(u + w) tiene degeneración:
- u(w) se ajusta para cancelar w
- Solo importa la suma u + w, no cada componente

SOLUCIÓN: FORMULACIÓN DE MOVIMIENTO INVERSO
--------------------------------------------
Basado en: Govindjee & Mihalic (1996), Sellier (2011)

En lugar de buscar una "predeformación w", formulamos:
- Configuración observada Ω_obs (de CT) como "actual"
- Configuración stress-free Ω_sf como "referencia" (desconocida)
- Mapeo inverso: φ⁻¹: Ω_obs → Ω_sf

Parametrización:
- Puntos en Ω_obs: x
- Puntos en Ω_sf: X = x - w(x)  donde w es el desplazamiento inverso
- Gradiente inverso: f = I - ∇w
- Gradiente forward: F = f⁻¹ = (I - ∇w)⁻¹

El problema inverso:
- Dado P_alv (presión), encontrar w tal que cuando aplicamos P_alv
  partiendo de Ω_sf, obtenemos Ω_obs

Condición clave:
- En Ω_sf sin cargas (P = 0), el esfuerzo debe ser cero: S(F=I) = 0
- Esto está garantizado por la formulación si Ψ(I) = 0 y ∂Ψ/∂E|_{E=0} = 0
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
from firedrake.adjoint import *
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# ============================================================
# 1. PARÁMETROS
# ============================================================

Phi_R = 0.74
R_RI = 0.11

c_fung = 2.5
a_fung = 0.433
b_fung = -0.61

gamma_0 = 70.0e-6
gamma_inf = 22.2e-6
gamma_min = 0.0
Gamma_inf = 3.0e-9
m1 = 47.8e-6
m2 = 140.0e-6
Gamma_max = Gamma_inf * (1.0 + (gamma_inf - gamma_min) / m2)

print("="*60)
print("FASE 2: FORMULACIÓN DE MOVIMIENTO INVERSO")
print("="*60)

# ============================================================
# 2. CARGAR DATOS
# ============================================================

input_dir = "phase1_results"
data = np.load(os.path.join(input_dir, "forward_data.npz"))

u_obs_data = data['u_data']
Gamma_obs_data = data['Gamma_data']
P_alv_value = data['P_alv_final']

print(f"\nDatos cargados:")
print(f"  P_alv = {P_alv_value*1000:.2f} Pa")

# ============================================================
# 3. MALLA EN CONFIGURACIÓN OBSERVADA
# ============================================================
"""
Interpretación clave:
- La malla representa la configuración OBSERVADA (deformada)
- Los nodos están en posiciones x = X_ref + u_obs
- Queremos encontrar X_ref (stress-free)
"""

Lx, Ly = 10.0, 10.0
nx, ny = 20, 20

# Esta malla representa Ω_obs
mesh_obs = fd.RectangleMesh(nx, ny, Lx, Ly)

V = fd.VectorFunctionSpace(mesh_obs, "CG", 2)
Q = fd.FunctionSpace(mesh_obs, "DG", 0)

dim = mesh_obs.geometric_dimension()
I = fd.Identity(dim)
x = fd.SpatialCoordinate(mesh_obs)

# Coordenadas de los nodos observados (incluyen deformación)
# En la malla real de pulmón, estos serían los voxels del CT

# ============================================================
# 4. INCÓGNITA: DESPLAZAMIENTO INVERSO
# ============================================================
"""
w(x): desplazamiento desde configuración observada a stress-free
X = x - w(x)

Gradiente del mapeo inverso:
f = ∂X/∂x = I - ∇w

Gradiente del mapeo forward (de stress-free a observada):
F = f⁻¹ = (I - ∇w)⁻¹
"""

w = fd.Function(V, name="inverse_displacement")
v_test = fd.TestFunction(V)

# Concentración de surfactante (observada)
Gamma = fd.Function(Q, name="Gamma")
Gamma.dat.data[:] = Gamma_obs_data

# Presión alveolar
P_alv = fd.Constant(P_alv_value)

# ============================================================
# 5. CINEMÁTICA INVERSA
# ============================================================

def inverse_kinematics(w_field):
    """
    Calcular gradiente de deformación F a partir del desplazamiento inverso w
    
    f = I - ∇w  (gradiente del mapeo inverso)
    F = f⁻¹     (gradiente del mapeo forward)
    """
    grad_w = fd.grad(w_field)
    f = I - grad_w
    
    # Regularización para evitar f singular
    det_f = fd.det(f)
    
    # F = f^{-1}
    F = fd.inv(f)
    
    return F, f, det_f

# ============================================================
# 6. MODELO CONSTITUTIVO
# ============================================================

def constitutive_model(F, Gamma_field, P_alv_const):
    """
    Calcular tensor de esfuerzos S dado F
    """
    J = fd.det(F)
    C = F.T * F
    invC = fd.inv(C)
    
    E = fd.variable(0.5 * (C - I))
    
    # Invariantes
    J1 = fd.tr(E)
    J2 = 0.5 * (fd.tr(E)**2 - fd.tr(E * E))
    
    # Fung
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
    S = S_el + (P_gamma - P_alv_const) * J * invC
    
    return S, J, gamma, P_gamma

# ============================================================
# 7. FORMULACIÓN VARIACIONAL
# ============================================================
"""
El equilibrio en la configuración de referencia es:
Div(F·S) = 0

Pero estamos trabajando en la configuración observada.
Necesitamos transformar la integral.

Si Ω_ref tiene elemento de volumen dV y Ω_obs tiene dv = J·dV:
∫_{Ω_ref} (F·S) : ∇δu dV = ∫_{Ω_obs} (F·S) : ∇δu · (1/J) dv

Pero aquí el gradiente es respecto a X (referencia), no x (observada).
∇_X = F^T · ∇_x

Integral transformada:
∫_{Ω_obs} (F·S) : (F^T · ∇_x δu) · det(f) dv

donde det(f) = 1/J transforma el elemento de volumen.
"""

def equilibrium_residual(w_field, Gamma_field, P_alv_const, test_func):
    """
    Residuo de equilibrio en formulación inversa
    """
    # Cinemática
    F, f, det_f = inverse_kinematics(w_field)
    
    # Constitutivo
    S, J, gamma, P_gamma = constitutive_model(F, Gamma_field, P_alv_const)
    
    # Primer tensor de Piola-Kirchhoff
    P = F * S
    
    # Gradiente de la función test respecto a config. de referencia
    # ∇_X v = F^T · ∇_x v
    grad_v_ref = F.T * fd.grad(test_func)
    
    # Residuo (integrado en config. observada, con factor de Jacobiano)
    # det(f) = 1/det(F) = 1/J es el Jacobiano inverso
    det_f_safe = fd.max_value(det_f, 1e-6)
    
    Residual = fd.inner(P, grad_v_ref) * det_f_safe * fd.dx
    
    return Residual, J, det_f

# ============================================================
# 8. CONDICIONES DE BORDE
# ============================================================
"""
Para el problema inverso, las BCs son diferentes:
- La malla observada YA está en su posición final
- Buscamos el mapeo a la referencia

Si un punto x está fijo en la observación, entonces w(x) = 0
(ese punto no se mueve entre configuraciones)
"""

bcs = [
    fd.DirichletBC(V, fd.Constant((0.0, 0.0)), 1),  # x = 0 fijo
    fd.DirichletBC(V.sub(1), 0.0, 3),               # y = 0, sin movimiento en y
]

# ============================================================
# 9. FUNCIONAL OBJETIVO
# ============================================================
"""
Queremos que el equilibrio se satisfaga con la presión observada.

Como estamos en formulación residual, el funcional es:
J(w) = ½||R(w)||² + regularización

donde R es el residuo de equilibrio.

Pero esto es cuadráticamente costoso. Mejor: resolvemos
R(w) = 0 directamente si es posible.

Alternativa: doble bucle
- Loop externo: optimizar w
- Loop interno: verificar consistencia física

Para el caso simple, tratamos de resolver el sistema no lineal directamente.
"""

# ============================================================
# 10. RESOLVER PROBLEMA INVERSO
# ============================================================

print("\n" + "="*60)
print("RESOLVIENDO PROBLEMA INVERSO")
print("="*60)

# Problema no lineal: encontrar w tal que equilibrio se satisface
# con la presión dada P_alv

Residual, J_det, det_f = equilibrium_residual(w, Gamma, P_alv, v_test)

# Solver no lineal
problem = fd.NonlinearVariationalProblem(Residual, w, bcs=bcs)

solver_params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_monitor": None,
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-6,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_params)

# Inicializar
w.assign(fd.Constant((0.0, 0.0)))

print("\nResolviendo sistema no lineal...")

try:
    solver.solve()
    print("✓ Convergió")
    success = True
except fd.ConvergenceError as e:
    print(f"✗ No convergió: {e}")
    success = False

# ============================================================
# 11. ANÁLISIS DE RESULTADOS
# ============================================================

if success:
    print("\n" + "="*60)
    print("ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    # Estadísticas del desplazamiento inverso
    w_norm = fd.norm(w)
    w_data = w.dat.data_ro
    
    print(f"\nDesplazamiento inverso w:")
    print(f"  ||w|| = {w_norm:.6e}")
    print(f"  w_x: [{w_data[:,0].min():.4e}, {w_data[:,0].max():.4e}]")
    print(f"  w_y: [{w_data[:,1].min():.4e}, {w_data[:,1].max():.4e}]")
    
    # Jacobiano del mapeo
    F_field, f_field, det_f_field = inverse_kinematics(w)
    J_proj = fd.project(fd.det(F_field), Q)
    det_f_proj = fd.project(det_f_field, Q)
    
    print(f"\nJacobiano F (forward):")
    print(f"  J: [{J_proj.dat.data_ro.min():.4f}, {J_proj.dat.data_ro.max():.4f}]")
    print(f"\nJacobiano f (inverse):")
    print(f"  det(f): [{det_f_proj.dat.data_ro.min():.4f}, {det_f_proj.dat.data_ro.max():.4f}]")
    
    # Verificar: sin presión, ¿el esfuerzo es cero?
    print("\n--- Verificación stress-free ---")
    
    # Con P_alv = 0
    P_zero = fd.Constant(0.0)
    S_check, J_check, _, _ = constitutive_model(F_field, Gamma, P_zero)
    S_norm = fd.project(fd.sqrt(fd.inner(S_check, S_check)), Q)
    
    print(f"  ||S|| con P=0: [{S_norm.dat.data_ro.min():.4e}, {S_norm.dat.data_ro.max():.4e}]")
    
    # Comparar con datos originales
    u_obs = fd.Function(V)
    u_obs.dat.data[:] = u_obs_data
    
    # La posición total observada debería ser x (coordenadas de la malla)
    # La posición en referencia es X = x - w
    # Entonces el desplazamiento u = x - X = w
    # Por lo tanto, w debería aproximar u_obs
    
    diff = fd.Function(V)
    diff.assign(w - u_obs)
    diff_norm = fd.norm(diff)
    u_obs_norm = fd.norm(u_obs)
    
    print(f"\nComparación con u_obs:")
    print(f"  ||w - u_obs|| = {diff_norm:.6e}")
    print(f"  ||u_obs|| = {u_obs_norm:.6e}")
    print(f"  Error relativo: {diff_norm/u_obs_norm*100:.2f}%")
    
    # ============================================================
    # 12. GUARDAR RESULTADOS
    # ============================================================
    
    output_dir = "phase2_results"
    os.makedirs(output_dir, exist_ok=True)
    
    from firedrake.output import VTKFile
    
    # Campos para visualización
    J_field = fd.Function(Q, name="Jacobian")
    J_field.assign(J_proj)
    
    # Configuración stress-free: X = x - w
    # Guardamos como campo de coordenadas transformadas
    X_stressfree = fd.Function(V, name="StressFree_Coords")
    
    # Interpolar coordenadas
    X_coord = fd.interpolate(fd.as_vector([x[0], x[1]]), V)
    X_stressfree.assign(X_coord - w)
    
    outfile = VTKFile(os.path.join(output_dir, "inverse_motion_result.pvd"))
    w.rename("inverse_displacement")
    u_obs.rename("observed_displacement")
    diff.rename("difference_w_minus_uobs")
    outfile.write(w, u_obs, diff, J_field, X_stressfree)
    
    # Datos numéricos
    np.savez(os.path.join(output_dir, "inverse_motion_data.npz"),
             w_data=w.dat.data_ro[:],
             J_data=J_field.dat.data_ro[:],
             error_rel=float(diff_norm/u_obs_norm))
    
    print(f"\nResultados guardados en: {output_dir}/")

else:
    print("\n⚠ El solver no convergió. Posibles causas:")
    print("  - Deformación demasiado grande")
    print("  - Singularidad en f = I - ∇w")
    print("  - Problema mal condicionado")
    print("\nIntentando con regularización incremental...")
    
    # Aquí podríamos implementar una continuación de parámetros

print("\n" + "="*60)
print("FASE 2 COMPLETADA")
print("="*60)
