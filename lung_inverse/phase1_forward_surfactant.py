"""
FASE 1: Problema Directo con Efectos de Surfactante
====================================================
Modelo simplificado 2D basado en Avilés-Rojas & Hurtado (2025)

Ecuaciones:
- Equilibrio: Div(F·S) = 0
- Tensor de esfuerzos: S = S_el + (P_γ - P_alv)·J·C⁻¹
- Presión de colapso: P_γ = (2γ/R_I)·(Φ_R/Φ)^(1/3)

Simplificaciones:
- 2D (estado plano de deformación)
- Quasi-estático (sin dinámica de Darcy)
- Surfactante en régimen insoluble (γ función de Γ)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import firedrake as fd
import numpy as np
from firedrake.output import VTKFile
import matplotlib.pyplot as plt

fd.parameters["form_compiler"]["quadrature_degree"] = 6

# ============================================================
# 1. PARÁMETROS DEL MODELO (Tabla 1 del paper)
# ============================================================

# Geometría alveolar
Phi_R = 0.74          # Porosidad de referencia [-]
R_RI = 0.11           # Radio interno de referencia [mm]

# Material Fung (tejido)
c_fung = 2.5       # [kPa] (paper usa 2.5 kPa, escalamos para 2D)
a_fung = 0.433        # [-]
b_fung = -0.61        # [-]

# Surfactante (Otis et al.)
gamma_0 = 70.0e-6     # Tensión superficial sin surfactante [N/mm = kPa·mm]
gamma_inf = 22.2e-6   # γ en umbral Langmuir/insoluble [N/mm]
gamma_min = 0.0       # Tensión mínima [N/mm]
Gamma_inf = 3.0e-9    # Concentración umbral [g/mm²]
m1 = 47.8e-6          # Pendiente régimen Langmuir [N/mm]
m2 = 140.0e-6         # Pendiente régimen insoluble [N/mm]

# Concentración máxima dinámica (Ec. 51)
Gamma_max = Gamma_inf * (1.0 + (gamma_inf - gamma_min) / m2)

print("="*60)
print("PARÁMETROS DEL MODELO")
print("="*60)
print(f"Φ_R = {Phi_R}, R_I = {R_RI} mm")
print(f"Fung: c={c_fung} kPa, a={a_fung}, b={b_fung}")
print(f"Surfactante: γ₀={gamma_0*1e6:.1f}, γ_inf={gamma_inf*1e6:.1f} μN/mm")
print(f"Γ_inf={Gamma_inf:.2e}, Γ_max={Gamma_max:.2e} g/mm²")

# ============================================================
# 2. MALLA Y ESPACIOS DE FUNCIONES
# ============================================================

# Dominio: lámina rectangular (representa parénquima)
Lx, Ly = 10.0, 10.0  # [mm]
nx, ny = 20, 20

mesh = fd.RectangleMesh(nx, ny, Lx, Ly)
print(f"\nMalla: {nx}x{ny} elementos, dominio {Lx}x{Ly} mm")

V = fd.VectorFunctionSpace(mesh, "CG", 2) # Desplazamiento
Q = fd.FunctionSpace(mesh, "DG", 0)       # Concentración surfactante (por elemento)

u = fd.Function(V, name="Displacement")
v = fd.TestFunction(V)
Gamma = fd.Function(Q, name="Surfactant_Concentration")

coords_initial = mesh.coordinates.dat.data_ro.copy()

dim = mesh.geometric_dimension()
I = fd.Identity(dim)

# ============================================================
# 3. MODELO CONSTITUTIVO
# ============================================================

# Cinemática
F = I + fd.grad(u)
J = fd.det(F)
C = F.T * F
invC = fd.inv(C)
# Tensor de Green-Lagrange
E = fd.variable(0.5 * (C - I))

# Invariantes de E
J1 = fd.tr(E)
J2 = 0.5 * (fd.tr(E)**2 - fd.tr(E * E))

# Energía elástica Fung (Ec. 44)
Psi_el = c_fung * (fd.exp(a_fung * J1**2 + b_fung * J2) - 1.0)
S_el = fd.diff(Psi_el, E)

# Porosidad Lagrangiana: Φ = J - 1 + Φ_R (Ec. 17)
Phi = fd.max_value(J - 1.0 + Phi_R, 1e-4)

# Tensión superficial según modelo de Otis (Ec. 48)
ratio = Gamma / Gamma_inf
gamma_lang = gamma_0 - m1 * ratio
gamma_insol = gamma_inf - m2 * (ratio - 1.0)
gamma = fd.conditional(fd.lt(Gamma, Gamma_inf), gamma_lang, gamma_insol)
gamma = fd.max_value(gamma, gamma_min)

# Presión de colapso: P_γ = (2γ/R_I)·(Φ_R/Φ)^(1/3) (Ec. 43)
P_gamma = (2.0 * gamma / R_RI) * ((Phi_R / Phi)**(1.0/3.0))

P_alv = fd.Constant(0.0)

# Tensor de esfuerzos total (Ec. 42) S = S_el + (P_γ - P_alv)·J·C⁻¹
S_total = S_el + (P_gamma - P_alv) * J * invC

# ============================================================
# 4. FORMULACIÓN VARIACIONAL
# ============================================================

# Residuo mecánico (Ec. 70)
F_stress = F * S_total
Residual = fd.inner(F_stress, fd.grad(v)) * fd.dx

bcs = [
    fd.DirichletBC(V, fd.Constant((0.0, 0.0)), 1),
    fd.DirichletBC(V.sub(1), 0.0, 3),
]

problem = fd.NonlinearVariationalProblem(Residual, u, bcs=bcs)

solver_params = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-6,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solver_params)

# ============================================================
# 5. SIMULACIÓN DE INFLACIÓN
# ============================================================

print("\n" + "="*60)
print("SIMULACIÓN DE INFLACIÓN")
print("="*60)

Gamma.assign(fd.Constant(Gamma_max))
u.assign(fd.Constant((0.0, 0.0)))

output_dir = "phase1_results"
os.makedirs(output_dir, exist_ok=True)

gamma_field = fd.Function(Q, name="Surface_Tension")
P_gamma_field = fd.Function(Q, name="Collapse_Pressure")
J_field = fd.Function(Q, name="Jacobian")

P_target = 1.0 # 1.0 kPa de carga máxima
n_steps = 40

steps_list = [0]
pressures = [0.0]
volumes = [0.0]
u_norms = [0.0]
energies = [0.0]

outfile = VTKFile(os.path.join(output_dir, "inflation.pvd"))

for step in range(1, n_steps + 1):
    P_val = P_target * step / n_steps
    P_alv.assign(P_val)
    
    print(f"\nStep {step}/{n_steps}: P_alv = {P_val:.4f} kPa")
    
    try:
        solver.solve()
    except fd.ConvergenceError:
        print("  ⚠ No convergió")
        break
    
    # Actualizar surfactante
    J_curr = fd.project(fd.det(I + fd.grad(u)), Q)
    Phi_curr = fd.project(fd.max_value(J_curr - 1.0 + Phi_R, 1e-4), Q)
    
    # Densidad de área interfacial: A = (3/R_I)·Φ_R^(1/3)·Φ^(2/3) (Ec. 39)
    # Ratio = (Phi_R / Phi)^(2/3)
    area_ratio = (Phi_R / Phi_curr)**(2.0/3.0)
    Gamma_new = fd.project(Gamma_max * area_ratio, Q)
    Gamma_new_data = np.clip(Gamma_new.dat.data[:], Gamma_inf, Gamma_max)
    Gamma.dat.data[:] = Gamma_new_data
    
    # Campos
    J_field.assign(J_curr)
    
    ratio_curr = Gamma / Gamma_inf
    gamma_expr = fd.conditional(fd.lt(Gamma, Gamma_inf), 
                                 gamma_0 - m1 * ratio_curr,
                                 gamma_inf - m2 * (ratio_curr - 1.0))
    gamma_expr = fd.max_value(gamma_expr, gamma_min)
    gamma_field.assign(fd.project(gamma_expr, Q))
    
    Phi_safe = fd.max_value(J_curr - 1.0 + Phi_R, 1e-4)
    P_gamma_expr = (2.0 * gamma_expr / R_RI) * ((Phi_R / Phi_safe)**(1.0/3.0))
    P_gamma_field.assign(fd.project(P_gamma_expr, Q))
    
    # Métricas
    V_change = fd.assemble((J_curr - 1.0) * fd.dx)
    u_L2 = fd.norm(u)
    energy = fd.assemble(Psi_el * fd.dx)
    
    steps_list.append(step)
    pressures.append(P_val)
    volumes.append(V_change)
    u_norms.append(u_L2)
    energies.append(energy)
    
    J_min, J_max = J_field.dat.data_ro.min(), J_field.dat.data_ro.max()
    gamma_avg = fd.assemble(gamma_field * fd.dx) / (Lx * Ly)
    
    print(f"  ||u||_L2 = {u_L2:.6e}, J ∈ [{J_min:.4f}, {J_max:.4f}]")
    print(f"  γ_avg = {gamma_avg*1e6:.2f} μN/mm, ΔV = {V_change:.4f} mm²")
    
    outfile.write(u, Gamma, gamma_field, P_gamma_field, J_field)

# ============================================================
# 6. GUARDAR MALLA DEFORMADA
# ============================================================

print("\n" + "="*60)
print("GUARDANDO RESULTADOS")
print("="*60)

V_coords = fd.VectorFunctionSpace(mesh, "CG", 1)
# Aquí saltará el warning de interpolate, es normal.
u_at_nodes = fd.interpolate(u, V_coords)
coords_deformed = coords_initial + u_at_nodes.dat.data_ro

mesh_deformed = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
mesh_deformed.coordinates.dat.data[:] = coords_deformed

deformed_file = VTKFile(os.path.join(output_dir, "mesh_deformed.pvd"))
V_def = fd.VectorFunctionSpace(mesh_deformed, "CG", 1)
zero_field = fd.Function(V_def, name="zero")
deformed_file.write(zero_field)

np.savez(os.path.join(output_dir, "forward_data.npz"),
         u_data=u.dat.data_ro[:],
         Gamma_data=Gamma.dat.data_ro[:],
         P_alv_final=float(P_alv),
         coords_initial=coords_initial,
         coords_deformed=coords_deformed,
         steps=np.array(steps_list),
         pressures=np.array(pressures),
         volumes=np.array(volumes),
         u_norms=np.array(u_norms),
         energies=np.array(energies))

# ============================================================
# 7. GRÁFICAS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(pressures, volumes, 'b-o', markersize=4)
axes[0, 0].set_xlabel('Presión [kPa]')
axes[0, 0].set_ylabel('ΔV [mm²]')
axes[0, 0].set_title('Curva P-V')
axes[0, 0].grid(True)

axes[0, 1].plot(steps_list, u_norms, 'r-o', markersize=4)
axes[0, 1].set_xlabel('Paso')
axes[0, 1].set_ylabel('||u||_L2 [mm]')
axes[0, 1].set_title('Norma L2 del Desplazamiento')
axes[0, 1].grid(True)

axes[1, 0].plot(steps_list[1:], energies[1:], 'g-o', markersize=4)
axes[1, 0].set_xlabel('Paso')
axes[1, 0].set_ylabel('Energía [kPa·mm²]')
axes[1, 0].set_title('Energía Elástica')
axes[1, 0].grid(True)

axes[1, 1].plot(coords_initial[:, 0], coords_initial[:, 1], 'b.', markersize=1, alpha=0.5, label='Inicial')
axes[1, 1].plot(coords_deformed[:, 0], coords_deformed[:, 1], 'r.', markersize=1, alpha=0.5, label='Deformado')
axes[1, 1].set_xlabel('x [mm]')
axes[1, 1].set_ylabel('y [mm]')
axes[1, 1].set_title('Configuración Inicial vs Deformada')
axes[1, 1].legend()
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "convergence_plots.png"), dpi=150)
plt.close()

# ============================================================
# 8. RESUMEN
# ============================================================

print(f"\nArchivos en: {output_dir}/")

displacement_magnitude = np.linalg.norm(coords_deformed - coords_initial, axis=1)

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"  Pasos: {len(steps_list)-1}/{n_steps}")
print(f"  P_alv final: {pressures[-1]:.4f} kPa")
print(f"  ||u||_L2 final: {u_norms[-1]:.6e} mm")
print(f"  ΔV final: {volumes[-1]:.4f} mm²")
print(f"  Desplazamiento máx: {displacement_magnitude.max():.6e} mm")
print(f"  Desplazamiento prom: {displacement_magnitude.mean():.6e} mm")

print("\nFASE 1 COMPLETADA")