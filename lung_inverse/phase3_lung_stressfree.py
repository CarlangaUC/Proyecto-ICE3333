"""
FASE 3: Aplicación a Pulmón Real desde CT
==========================================

Este script muestra cómo aplicar la metodología de recuperación
de configuración stress-free a una malla de pulmón real.

Flujo de trabajo:
1. Cargar malla del pulmón (desde CT)
2. Definir condiciones de borde anatómicas
3. Estimar parámetros de presión pleural
4. Resolver problema inverso
5. Obtener configuración stress-free

Nota: La malla del pulmón debe estar en formato compatible con Firedrake
(por ejemplo, Gmsh .msh)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
from firedrake.adjoint import *
import numpy as np
from firedrake.output import VTKFile

# ============================================================
# 1. PARÁMETROS DEL MODELO
# ============================================================

# Geometría alveolar (Tabla 1 del paper)
Phi_R = 0.74          # Porosidad de referencia [-]
R_RI = 0.11           # Radio interno de referencia [mm]

# Material Fung
c_fung = 2.5          # [kPa] - valor original del paper
a_fung = 0.433        # [-]
b_fung = -0.61        # [-]

# Surfactante
gamma_0 = 70.0e-6     # [N/mm]
gamma_inf = 22.2e-6   # [N/mm]
gamma_min = 0.0       # [N/mm]
Gamma_inf = 3.0e-9    # [g/mm²]
m1 = 47.8e-6          # [N/mm]
m2 = 140.0e-6         # [N/mm]
Gamma_max = Gamma_inf * (1.0 + (gamma_inf - gamma_min) / m2)

# Presión pleural típica en reposo
P_pleural = -0.5      # [kPa] ~ -5 cm H2O (negativa = succión)

print("="*60)
print("FASE 3: RECUPERACIÓN STRESS-FREE EN PULMÓN REAL")
print("="*60)

# ============================================================
# 2. CARGAR MALLA DEL PULMÓN
# ============================================================
"""
En la práctica, la malla vendría de:
1. Segmentación de CT
2. Generación de malla con CGAL, TetGen, o Gmsh
3. Conversión a formato compatible

Aquí usamos una malla simplificada como ejemplo.
Para malla real, usar:
    mesh = fd.Mesh("lung_mesh.msh")
"""

def create_example_lung_mesh():
    """
    Crear malla simplificada tipo pulmón (elipsoide)
    En práctica, cargar desde archivo.
    """
    # Dimensiones aproximadas de un pulmón
    Lx, Ly, Lz = 120.0, 150.0, 200.0  # mm
    nx, ny, nz = 10, 12, 16
    
    # Malla rectangular (simplificación)
    mesh = fd.BoxMesh(nx, ny, nz, Lx, Ly, Lz)
    
    return mesh, Lx, Ly, Lz

# Cargar o crear malla
print("\nCargando malla del pulmón...")

# Opción 1: Malla de ejemplo
mesh, Lx, Ly, Lz = create_example_lung_mesh()

# Opción 2: Cargar malla real (descomentar si disponible)
# mesh = fd.Mesh("lung_mesh.msh")

n_cells = mesh.num_cells()
n_vertices = mesh.num_vertices()
print(f"  Malla cargada: {n_cells} elementos, {n_vertices} vértices")

# ============================================================
# 3. ESPACIOS DE FUNCIONES
# ============================================================

dim = mesh.geometric_dimension()
I = fd.Identity(dim)
x = fd.SpatialCoordinate(mesh)

# Espacios
V = fd.VectorFunctionSpace(mesh, "CG", 1)  # Grado 1 para eficiencia en 3D
Q = fd.FunctionSpace(mesh, "DG", 0)

print(f"  Espacio V: {V.dim()} DoFs")
print(f"  Espacio Q: {Q.dim()} DoFs")

# ============================================================
# 4. CONDICIONES DE BORDE ANATÓMICAS
# ============================================================
"""
Bordes del pulmón:
1. Vías aéreas (bronquios principales) - presión de aire
2. Pleura visceral - presión pleural
3. Hilio pulmonar - fijo (conexión con mediastino)

Para el problema inverso:
- Los puntos cerca del hilio están fijos (w = 0)
- La pleura está sujeta a presión pleural
"""

def define_anatomical_regions(mesh, Lx, Ly, Lz):
    """
    Definir regiones anatómicas para condiciones de borde
    
    En malla real, esto vendría de etiquetas del mallador
    """
    x = fd.SpatialCoordinate(mesh)
    
    # Región del hilio (cerca de y = Ly, z ≈ Lz/2)
    # En malla real: marcador específico
    hilum_region = fd.conditional(
        fd.And(fd.gt(x[1], 0.9*Ly), 
               fd.And(fd.gt(x[2], 0.4*Lz), fd.lt(x[2], 0.6*Lz))),
        1.0, 0.0
    )
    
    return hilum_region

hilum_marker = define_anatomical_regions(mesh, Lx, Ly, Lz)

# Condiciones de borde
# Para malla real con marcadores:
#   bcs = [fd.DirichletBC(V, Constant((0,0,0)), "hilum")]

# Para malla de ejemplo, fijamos una cara
bcs = [
    fd.DirichletBC(V, fd.Constant((0.0, 0.0, 0.0)), 2),  # y = Ly (hilio)
]

# ============================================================
# 5. VARIABLES DEL PROBLEMA INVERSO
# ============================================================

# Desplazamiento inverso (incógnita)
w = fd.Function(V, name="inverse_displacement")
v_test = fd.TestFunction(V)

# Concentración de surfactante (asumida uniforme inicialmente)
Gamma = fd.Function(Q, name="surfactant")
Gamma.assign(fd.Constant(0.5 * (Gamma_inf + Gamma_max)))  # Estado intermedio

# Presión
P_alv = fd.Constant(-P_pleural)  # Presión transmural

print(f"\nCondiciones del problema:")
print(f"  P_pleural = {P_pleural} kPa")
print(f"  Γ inicial = {float(Gamma.dat.data_ro.mean()):.2e} g/mm²")

# ============================================================
# 6. MODELO CONSTITUTIVO 3D
# ============================================================

def inverse_kinematics_3d(w_field):
    """Cinemática inversa en 3D"""
    grad_w = fd.grad(w_field)
    f = I - grad_w
    det_f = fd.det(f)
    F = fd.inv(f)
    return F, f, det_f

def constitutive_model_3d(F, Gamma_field, P_alv_const):
    """Modelo constitutivo 3D con surfactante"""
    J = fd.det(F)
    C = F.T * F
    invC = fd.inv(C)
    
    E = fd.variable(0.5 * (C - I))
    
    # Invariantes 3D
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

def equilibrium_residual_3d(w_field, Gamma_field, P_alv_const, test_func):
    """Residuo de equilibrio en 3D"""
    F, f, det_f = inverse_kinematics_3d(w_field)
    S, J, gamma, P_gamma = constitutive_model_3d(F, Gamma_field, P_alv_const)
    
    P = F * S
    grad_v_ref = F.T * fd.grad(test_func)
    det_f_safe = fd.max_value(det_f, 1e-6)
    
    Residual = fd.inner(P, grad_v_ref) * det_f_safe * fd.dx
    
    return Residual

Residual = equilibrium_residual_3d(w, Gamma, P_alv, v_test)

# ============================================================
# 8. SOLVER
# ============================================================

problem = fd.NonlinearVariationalProblem(Residual, w, bcs=bcs)

solver_params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",  # Backtracking para 3D
    "snes_monitor": None,
    "snes_max_it": 50,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-5,
    "snes_stol": 1e-8,
    "ksp_type": "gmres",
    "ksp_max_it": 200,
    "pc_type": "ilu",  # ILU para problemas grandes
    # Para problemas muy grandes, usar:
    # "pc_type": "hypre",
    # "pc_hypre_type": "boomeramg",
}

solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_params)

# ============================================================
# 9. RESOLVER CON CONTINUACIÓN DE CARGA
# ============================================================

print("\n" + "="*60)
print("RESOLVIENDO PROBLEMA INVERSO 3D")
print("="*60)

# Inicializar
w.assign(fd.Constant((0.0, 0.0, 0.0)))

# Continuación de carga para mejor convergencia
n_steps = 5
P_final = float(P_alv)

print(f"\nContinuación de carga en {n_steps} pasos...")

output_dir = "phase3_results"
os.makedirs(output_dir, exist_ok=True)

success = True
for step in range(1, n_steps + 1):
    # Incrementar presión gradualmente
    P_current = P_final * step / n_steps
    P_alv.assign(P_current)
    
    print(f"\n  Paso {step}/{n_steps}: P = {P_current:.4f} kPa")
    
    try:
        solver.solve()
        
        # Estadísticas
        w_norm = fd.norm(w)
        F_field, _, det_f_field = inverse_kinematics_3d(w)
        J_val = fd.assemble(fd.det(F_field) * fd.dx) / fd.assemble(fd.Constant(1.0) * fd.dx)
        
        print(f"    ✓ Convergió: ||w|| = {w_norm:.4e}, J_avg = {J_val:.4f}")
        
    except fd.ConvergenceError:
        print(f"    ✗ No convergió")
        success = False
        break

# ============================================================
# 10. POST-PROCESAMIENTO
# ============================================================

if success:
    print("\n" + "="*60)
    print("ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    # Campos derivados
    F_field, f_field, det_f_field = inverse_kinematics_3d(w)
    
    J_field = fd.Function(Q, name="Jacobian")
    J_field.assign(fd.project(fd.det(F_field), Q))
    
    det_f_proj = fd.Function(Q, name="det_f")
    det_f_proj.assign(fd.project(det_f_field, Q))
    
    print(f"\nEstadísticas del mapeo:")
    print(f"  ||w|| = {fd.norm(w):.4e} mm")
    print(f"  J (forward): [{J_field.dat.data_ro.min():.4f}, {J_field.dat.data_ro.max():.4f}]")
    print(f"  det(f): [{det_f_proj.dat.data_ro.min():.4f}, {det_f_proj.dat.data_ro.max():.4f}]")
    
    # Cambio de volumen total
    V_ref = fd.assemble(det_f_field * fd.dx)  # Volumen de referencia
    V_obs = fd.assemble(fd.Constant(1.0) * fd.dx)  # Volumen observado
    
    print(f"\nVolúmenes:")
    print(f"  V_observado = {V_obs:.2f} mm³ = {V_obs/1e6:.4f} L")
    print(f"  V_stress-free = {V_ref:.2f} mm³ = {V_ref/1e6:.4f} L")
    print(f"  Ratio: {V_obs/V_ref:.4f}")
    
    # Verificación stress-free
    print("\n--- Verificación stress-free (P = 0) ---")
    P_zero = fd.Constant(0.0)
    S_check, _, _, _ = constitutive_model_3d(F_field, Gamma, P_zero)
    S_norm = fd.project(fd.sqrt(fd.inner(S_check, S_check)), Q)
    
    print(f"  ||S|| con P=0: [{S_norm.dat.data_ro.min():.4e}, {S_norm.dat.data_ro.max():.4e}]")
    
    # ============================================================
    # 11. CREAR MALLA STRESS-FREE
    # ============================================================
    """
    Para crear la malla de la configuración stress-free:
    1. X = x - w(x) para cada nodo
    2. Actualizar coordenadas de la malla
    """
    
    print("\n--- Creando malla stress-free ---")
    
    # Obtener coordenadas de los nodos
    coords_obs = mesh.coordinates.dat.data_ro.copy()
    
    # Interpolar w en los nodos de la malla
    V_coords = fd.VectorFunctionSpace(mesh, "CG", 1)
    w_interp = fd.interpolate(w, V_coords)
    w_at_nodes = w_interp.dat.data_ro
    
    # Calcular coordenadas stress-free
    coords_stressfree = coords_obs - w_at_nodes
    
    print(f"  Coordenadas observadas: x ∈ [{coords_obs.min():.2f}, {coords_obs.max():.2f}]")
    print(f"  Coordenadas stress-free: X ∈ [{coords_stressfree.min():.2f}, {coords_stressfree.max():.2f}]")
    
    # Crear nueva malla con coordenadas transformadas
    # (En Firedrake, modificar coordenadas directamente en la malla)
    mesh_stressfree = fd.Mesh(mesh.coordinates.copy())
    mesh_stressfree.coordinates.dat.data[:] = coords_stressfree
    
    # ============================================================
    # 12. GUARDAR RESULTADOS
    # ============================================================
    
    # Guardar malla observada con campos
    outfile_obs = VTKFile(os.path.join(output_dir, "lung_observed.pvd"))
    w.rename("inverse_displacement")
    outfile_obs.write(w, J_field, S_norm)
    
    # Guardar malla stress-free
    V_sf = fd.VectorFunctionSpace(mesh_stressfree, "CG", 1)
    w_sf = fd.Function(V_sf, name="zero_displacement")  # w=0 en config stress-free
    
    outfile_sf = VTKFile(os.path.join(output_dir, "lung_stressfree.pvd"))
    outfile_sf.write(w_sf)
    
    # Datos numéricos
    np.savez(os.path.join(output_dir, "lung_inverse_data.npz"),
             w_data=w.dat.data_ro[:],
             coords_obs=coords_obs,
             coords_stressfree=coords_stressfree,
             J_data=J_field.dat.data_ro[:],
             V_obs=V_obs,
             V_ref=V_ref)
    
    print(f"\nResultados guardados en: {output_dir}/")
    print("  - lung_observed.pvd: malla observada con campos")
    print("  - lung_stressfree.pvd: malla stress-free")
    print("  - lung_inverse_data.npz: datos numéricos")

else:
    print("\n⚠ El solver no convergió.")
    print("Sugerencias:")
    print("  1. Reducir la presión inicial")
    print("  2. Aumentar número de pasos de continuación")
    print("  3. Refinar la malla")
    print("  4. Ajustar parámetros del material")

# ============================================================
# 13. RESUMEN Y SIGUIENTE PASOS
# ============================================================

print("\n" + "="*60)
print("FASE 3 COMPLETADA")
print("="*60)

print("""
RESUMEN DEL FLUJO DE TRABAJO:
============================

1. FASE 1: Problema directo en lámina 2D
   - Validar modelo con surfactante
   - Generar datos sintéticos
   
2. FASE 2: Problema inverso en lámina 2D  
   - Recuperar configuración stress-free
   - Validar metodología de movimiento inverso
   
3. FASE 3: Aplicación a pulmón 3D
   - Cargar malla de CT
   - Resolver problema inverso
   - Obtener geometría stress-free

APLICACIONES:
=============
- Simulación de ventilación mecánica desde config. real
- Análisis de esfuerzos pulmonares
- Planificación de tratamientos
- Modelado personalizado

LIMITACIONES:
=============
- Asume material homogéneo
- No incluye árbol bronquial explícito
- Surfactante simplificado (sin cinética Langmuir completa)
- Requiere estimación de presión pleural
""")
