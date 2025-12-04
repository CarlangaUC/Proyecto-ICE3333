# Recuperación de Configuración Stress-Free en Pulmón

## Descripción

Este proyecto implementa un flujo de trabajo para recuperar la configuración libre de tensiones (stress-free) de un pulmón a partir de imágenes de CT, basado en la formulación poromecánica con surfactante de Avilés-Rojas & Hurtado (2025).

## Estructura del Proyecto

```
lung_inverse/
├── README.md                    # Este archivo
├── phase1_forward_surfactant.py # Problema directo con surfactante
├── phase2_inverse_motion.py     # Problema inverso (formulación correcta)
├── phase2_inverse_stressfree.py # Versión alternativa (con limitaciones)
├── phase3_lung_stressfree.py    # Aplicación a pulmón 3D real
├── phase1_results/              # Resultados Fase 1
├── phase2_results/              # Resultados Fase 2
└── phase3_results/              # Resultados Fase 3
```

## Fundamento Teórico

### Modelo Constitutivo (Paper JMPS 2025)

El tensor de esfuerzos del parénquima pulmonar es:

$$\mathbf{S} = \frac{\partial \Psi^{el}}{\partial \mathbf{E}} + P_\gamma J \mathbf{C}^{-1} - P_{alv} J \mathbf{C}^{-1}$$

donde:
- $\Psi^{el}$: Energía elástica Fung
- $P_\gamma = \frac{2\gamma}{R_I}\left(\frac{\Phi_R}{\Phi}\right)^{1/3}$: Presión de colapso
- $\gamma$: Tensión superficial (depende del surfactante)
- $P_{alv}$: Presión alveolar

### Formulación de Movimiento Inverso

Para recuperar la configuración stress-free, usamos:

- **Configuración observada** $\Omega_{obs}$: La malla de CT
- **Configuración stress-free** $\Omega_{sf}$: La que buscamos
- **Mapeo inverso**: $\boldsymbol{\phi}^{-1}: \Omega_{obs} \to \Omega_{sf}$

Parametrización:
$$\mathbf{X} = \mathbf{x} - \mathbf{w}(\mathbf{x})$$

donde $\mathbf{w}$ es el desplazamiento inverso (incógnita).

Gradientes:
- Inverso: $\mathbf{f} = \mathbf{I} - \nabla\mathbf{w}$
- Forward: $\mathbf{F} = \mathbf{f}^{-1}$

## Flujo de Trabajo

### Fase 1: Validación del Modelo Directo

```bash
python phase1_forward_surfactant.py
```

**Objetivo**: Verificar que el modelo con surfactante funciona correctamente.

**Entrada**: Parámetros del material (Tabla 1 del paper)

**Salida**:
- `phase1_results/inflation.pvd`: Evolución temporal
- `phase1_results/deformed_state.pvd`: Estado final
- `phase1_results/forward_data.npz`: Datos para Fase 2

### Fase 2: Problema Inverso en 2D

```bash
python phase2_inverse_motion.py
```

**Objetivo**: Recuperar configuración stress-free a partir del estado deformado.

**Entrada**: Datos de Fase 1

**Salida**:
- `phase2_results/inverse_motion_result.pvd`: Desplazamiento inverso
- Verificación: $\|\mathbf{w} - \mathbf{u}_{obs}\|$ debe ser pequeño

### Fase 3: Aplicación a Pulmón 3D

```bash
python phase3_lung_stressfree.py
```

**Objetivo**: Obtener geometría stress-free de pulmón real.

**Entrada**: 
- Malla de CT (`lung_mesh.msh`)
- Presión pleural estimada

**Salida**:
- `phase3_results/lung_observed.pvd`: Malla original con campos
- `phase3_results/lung_stressfree.pvd`: Malla stress-free
- `phase3_results/lung_inverse_data.npz`: Coordenadas y datos

## Parámetros del Modelo

| Parámetro | Valor | Unidades | Descripción |
|-----------|-------|----------|-------------|
| $\Phi_R$ | 0.74 | - | Porosidad de referencia |
| $R_{RI}$ | 0.11 | mm | Radio interno alveolar |
| $c$ | 2.5 | kPa | Módulo Fung |
| $a$ | 0.433 | - | Parámetro Fung |
| $b$ | -0.61 | - | Parámetro Fung |
| $\gamma_0$ | 70×10⁻⁶ | N/mm | Tensión superficial máxima |
| $\gamma_{min}$ | 0 | N/mm | Tensión superficial mínima |
| $\Gamma_\infty$ | 3×10⁻⁹ | g/mm² | Concentración umbral |

## Requisitos

- Python 3.8+
- Firedrake (con adjoint)
- NumPy, SciPy
- ParaView (para visualización)

### Instalación de Firedrake

```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --install pyadjoint
source firedrake/bin/activate
```

## Notas Importantes

### Sobre la Identificabilidad

La formulación aditiva $\mathbf{F} = \mathbf{I} + \nabla(\mathbf{u} + \mathbf{w})$ tiene problemas de degeneración porque $\mathbf{u}$ y $\mathbf{w}$ no son separables. 

Por eso usamos la **formulación de movimiento inverso**:
- $\mathbf{F} = (\mathbf{I} - \nabla\mathbf{w})^{-1}$
- El problema es resolver el equilibrio directamente para $\mathbf{w}$
- No hay ambigüedad entre desplazamiento elástico y predeformación

### Limitaciones

1. **Homogeneidad**: El modelo asume propiedades uniformes
2. **Surfactante simplificado**: Solo regímenes Langmuir e insoluble
3. **Sin árbol bronquial**: Vías aéreas no modeladas explícitamente
4. **Presión pleural**: Debe estimarse o medirse

### Mejoras Futuras

- [ ] Incorporar heterogeneidad espacial de parámetros
- [ ] Añadir cinética completa del surfactante (Ec. 47)
- [ ] Acoplar con modelo de vías aéreas
- [ ] Validación con datos experimentales

## Referencias

1. Avilés-Rojas, N., & Hurtado, D.E. (2025). Integrating pulmonary surfactant into lung mechanical simulations: A continuum approach to surface tension in poromechanics. *J. Mech. Phys. Solids*, 203, 106174.

2. Govindjee, S., & Mihalic, P.A. (1996). Computational methods for inverse finite elastostatics. *Comput. Methods Appl. Mech. Engrg.*, 136, 47-57.

3. Sellier, M. (2011). An iterative method for the inverse elasto-static problem. *J. Fluids Struct.*, 27, 1461-1470.

## Autor

Código desarrollado para investigación en biomecánica pulmonar computacional.

## Licencia

MIT License
