from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import (Identity, TestFunctions, TrialFunctions, split, 
                 grad, det, tr, inner, derivative, dx, variable, inv, ln, TrialFunction)
import ufl
import os
import tempfile
import pickle
from basix.cell import CellType

# Cargar Malla! (Anillo)
MESH_FILENAME = "simple_annulus_mesh.xdmf"

try:
    with io.XDMFFile(MPI.COMM_WORLD, MESH_FILENAME, "r") as xdmf:
        domain = xdmf.read_mesh()
    print(f"✓ Malla cargada: {MESH_FILENAME}")
    
    R_in = 0.1
    R_out = 1.0

except Exception as e:
    print(f"❌ Error al cargar {MESH_FILENAME}: {e}. Generando malla de SLAB de respaldo.")
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
    R_in = 0.0 # Parámetros ajustados para el SLAB
    R_out = 1.0


# ELEMENTOS ROBUSTOS
k = 2
V_el = element("Lagrange", domain.basix_cell(), k, shape=(domain.geometry.dim,))
Q_el = element("Lagrange", domain.basix_cell(), k-1)
V_mixed = mixed_element([V_el, Q_el])
V = fem.functionspace(domain, V_mixed)

# PARÁMETROS 
mu = fem.Constant(domain, 1.0)
lmbda = fem.Constant(domain, 100.0)

# Función para la solución
u_p_ = fem.Function(V)
u, p = split(u_p_)

# Funciones test
v, q = TestFunctions(V)

# --- Neo-Hooke mixto (u, p) ---

d = domain.geometry.dim
I = Identity(d)

# Desplazamiento u y presión p ya definidos como Function en su espacio
# u_p_ = Function(W) con split(u_p_) -> (u, p)

F = ufl.variable(I + grad(u))
J = ufl.det(F)
C = F.T * F
Ic = ufl.tr(C)

# Parte isocórica (deviatoric) tipo Neo-Hooke modificado
psi_iso = (mu / 2) * (Ic - 2 - 2 * ufl.ln(J))

# Parte volumétrica en formulación mixta
psi_vol = -p * ufl.ln(J) + (1.0 / (2.0 * lmbda)) * p**2

psi = psi_iso + psi_vol

# Piola de 1ª especie
P = ufl.diff(psi, F)

# Forma débil: equilibrio + ecuación de estado (ln J - p/λ = 0)
G = (
    inner(P, grad(v)) * dx
    + (ufl.ln(J) - p / lmbda) * q * dx
)


# Jacobiano para Newton
du = TrialFunction(V)          # V: espacio de desplazamientos dentro de W
J_form = derivative(G, u_p_, du)


# CONDICIONES DE BORDE, Readaptar considerando cuerpos complejos posiblemente, por ahora anillo

def inner_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    # R_in es 0.1 en la malla de anillo.
    return np.isclose(r, R_in)

def outer_boundary(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    # R_out es 1.0 en la malla de anillo.
    return np.isclose(r, R_out)

inner_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, inner_boundary)
outer_facets = mesh.locate_entities_boundary(domain, domain.topology.dim-1, outer_boundary)

# Fija u_x = 0 y u_y = 0 en la cara exterior!
V0_outer, _ = V.sub(0).collapse()
outer_dofs = fem.locate_dofs_topological((V.sub(0), V0_outer), domain.topology.dim-1, outer_facets)

u_outer = fem.Function(V0_outer)
u_outer.x.array[:] = 0.0
bc_outer = fem.dirichletbc(u_outer, outer_dofs, V.sub(0))

# Presión alveolar aplicada (Carga de Neumann)
p_alv_const = fem.Constant(domain, 0.0)

# Etiquetar la frontera interna
fdim = domain.topology.dim - 1
marked_facets = np.hstack([inner_facets])
marked_values = np.hstack([np.full_like(inner_facets, 1)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# Agregar término de presión (Neumann) a la forma débil
n = ufl.FacetNormal(domain)
G += inner(-p_alv_const * J * inv(F).T * n, v) * ds(1)

bcs = [bc_outer]
print("✓ BCs corregidas: Borde externo fijo (Dirichlet). Presión interna (Neumann) aplicada en borde interno.")

# SOLVER CONFIGURADO
problem = NonlinearProblem(G, u_p_, bcs=bcs, J=J_form)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.max_it = 50
solver.convergence_criterion = "incremental"

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# CARGA ADAPTATIVA
def adaptive_load_increments(initial_disp, target_disp, max_steps=50):
    increments = []
    current = initial_disp
    
    while current < target_disp:
        increments.append(current)
        if current < target_disp * 0.1:
            step = target_disp / 30
        elif current < target_disp * 0.5:
            step = target_disp / 20
        else:
            step = target_disp / 15
            
        current += step
        if current > target_disp:
            current = target_disp
    
    increments.append(target_disp)
    return np.unique(increments)

initial_pressure = 0.01
target_pressure = 1.0
max_steps = 100

load_values = adaptive_load_increments(initial_pressure, target_pressure, max_steps)
num_steps = len(load_values) - 1

print(f"Estrategia de carga adaptativa:")
print(f"   - Presión objetivo: {target_pressure:.3f}")
print(f"   - Número de pasos: {num_steps}")

# Preparar espacios de visualización P1
V_vis = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
Q_vis = fem.functionspace(domain, ("Lagrange", 1))

print("\n=== Iniciando simulación directa ===")

convergence_history = []
all_converged = True

for i, pressure in enumerate(load_values[1:]):
    step_size = pressure - load_values[i]
    print(f"Paso {i+1}/{num_steps}, Presión = {pressure:.4f} (Δ = {step_size:.4f})")
    
    p_alv_const.value = pressure
    
    try:
        num_its, converged = solver.solve(u_p_)
        
        if converged:
            print(f"   ✓ Converged in {num_its} iterations")
            convergence_history.append(num_its)
            
            u_sol = u_p_.sub(0).collapse()
            u_mag = fem.Function(Q_vis)
            u_mag.interpolate(fem.Expression(ufl.sqrt(inner(u_sol, u_sol)), Q_vis.element.interpolation_points()))
            max_deformation = np.max(u_mag.x.array)

            print(f"   → Deformación máxima (Magnitud): {max_deformation:.4f}")
            
        else:
            print(f"   ✗ FAILED to converge after {num_its} iterations")
            all_converged = False
            break
            
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        all_converged = False
        break

print("\n=== Guardando resultados ===")

if all_converged:
    
    print("--- Guardando datos para problema inverso (Alta fidelidad) ---")
    
    u_sol = u_p_.sub(0).collapse()
    p_sol = u_p_.sub(1).collapse()
    u_sol.name = "Displacement"
    p_sol.name = "Pressure"

    # --- 2. DATOS PARA VISUALIZACIÓN EN XDMF (P1) ---
    print("\n--- Guardando resultados en XDMF para visualización (P1) ---")
    
    # Interpolar a P1 para XDMF
    u_vis = fem.Function(V_vis)
    u_vis.name = "Displacement"
    u_vis.interpolate(u_sol)
    
    p_vis = fem.Function(Q_vis)
    p_vis.name = "Pressure"
    p_vis.interpolate(p_sol)
    
    # Guardar en XDMF (malla original)
    xdmf_file = io.XDMFFile(domain.comm, os.path.join("simulation_results.xdmf"), "w")
    xdmf_file.write_mesh(domain)
    xdmf_file.write_function(u_vis, 0.0)
    xdmf_file.write_function(p_vis, 0.0)
    xdmf_file.close()
    print("✓ XDMF guardado: simulation_results.xdmf")
    
    print("\n--- Creando malla deformada para visualización ---")
        
    # 1. Calcular nuevas coordenadas
    original_coords = domain.geometry.x
    displacement_at_vertices = u_vis.x.array.reshape((-1, domain.geometry.dim))
    
    displacement_3d = np.zeros_like(original_coords)
    displacement_3d[:, :domain.geometry.dim] = displacement_at_vertices
    new_coords = original_coords + displacement_3d
    
    # 2. CORRECCIÓN DE CONECTIVIDAD (Sin usar basix explícitamente)
    # Obtenemos la topología de la malla actual
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0) # Generar mapa Celdas -> Vértices
    connectivity = domain.topology.connectivity(tdim, 0)
    
    # Contamos cuántos vértices tiene la primera celda para saber el tamaño (ej. 3 para triángulos)
    num_verts_per_cell = connectivity.links(0).size
    
    # Reconstruimos el array de celdas
    cells_array = connectivity.array.reshape((-1, num_verts_per_cell))
    cells_array = np.asarray(cells_array, dtype=np.int64)
    
    # 3. Crear la malla deformada
    deformed_mesh = mesh.create_mesh(domain.comm, cells_array, new_coords[:, :domain.geometry.dim], domain.ufl_domain())
    
    # Crear funciones en malla deformada para guardar los datos
    V_vis_deformed = fem.functionspace(deformed_mesh, ("Lagrange", 1, (deformed_mesh.geometry.dim,)))
    Q_vis_deformed = fem.functionspace(deformed_mesh, ("Lagrange", 1))
    
    u_vis_def = fem.Function(V_vis_deformed)
    u_vis_def.name = "Displacement"
    u_vis_def.x.array[:] = u_vis.x.array
    
    p_vis_def = fem.Function(Q_vis_deformed)
    p_vis_def.name = "Pressure"
    p_vis_def.x.array[:] = p_vis.x.array
    
    # Guardar malla deformada en XDMF
    xdmf_deformed = io.XDMFFile(deformed_mesh.comm, os.path.join("deformed_results.xdmf"), "w")
    xdmf_deformed.write_mesh(deformed_mesh)
    xdmf_deformed.write_function(u_vis_def, 0.0)
    xdmf_deformed.write_function(p_vis_def, 0.0)
    xdmf_deformed.close()
    print("✓ XDMF deformado guardado: deformed_results.xdmf")

else:
    print(f"✗ Simulación no convergente. No se guardaron resultados en: deformed_results.xdmf")