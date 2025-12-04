# Recuperación de Configuración Stress-Free en Pulmón

## Descripción General

Este proyecto implementa un marco computacional para resolver el **Problema Inverso Geométrico** (Inverse Elastostatics) en biomecánica pulmonar. El objetivo principal es recuperar la configuración de referencia libre de tensiones (*stress-free*) del parénquima pulmonar a partir de una geometría deformada (obtenida mediante tomografía computarizada - CT) y condiciones de carga conocidas (presión alveolar y distribución de surfactante).

El modelo incorpora la física avanzada del **surfactante pulmonar** dentro de un marco de poromecánica finita, basándose en la formulación termodinámicamente consistente de Avilés-Rojas & Hurtado (2025).

## 1. Formulación Constitutiva (Modelo Directo)

El comportamiento del parénquima se modela como un medio poroso saturado sometido a grandes deformaciones. El Tensor de Esfuerzos de Segundo Piola-Kirchhoff total ($\mathbf{S}$) se descompone en tres contribuciones:

$$\mathbf{S}_{total} = \mathbf{S}_{el} + \mathbf{S}_{surf} + \mathbf{S}_{air}$$

### A. Elasticidad del Tejido ($\mathbf{S}_{el}$)
Se utiliza un modelo hiperelástico exponencial (tipo Fung) para la matriz de tejido conectivo:
$$\Psi^{el} = c \left( \exp(a J_1^2 + b J_2) - 1 \right)$$
$$\mathbf{S}_{el} = 2 \frac{\partial \Psi^{el}}{\partial \mathbf{C}}$$

### B. Tensión Superficial y Surfactante ($\mathbf{S}_{surf}$)
Esta es la contribución principal que diferencia este modelo de la elasticidad estándar. Representa la presión de colapso generada por la interfaz aire-líquido:
$$\mathbf{S}_{surf} = P_\gamma J \mathbf{C}^{-1}$$

Donde la presión de colapso macroscópica $P_\gamma$ se deriva de la Ley de Laplace generalizada para medios porosos:
$$P_\gamma = \frac{2\gamma(\Gamma)}{R_{RI}} \left( \frac{\Phi_R}{\Phi} \right)^{1/3}$$

* $\gamma(\Gamma)$: Tensión superficial dependiente de la concentración de surfactante (Ec. de estado).
* $\Phi$: Porosidad actual (fracción de volumen de aire).
* $R_{RI}$: Radio alveolar representativo.

### C. Presión del Aire ($\mathbf{S}_{air}$)
La presión alveolar ($P_{alv}$) actúa hidrostáticamente sobre el volumen de poro deformado:
$$\mathbf{S}_{air} = - P_{alv} J \mathbf{C}^{-1}$$

---

## 2. Formulación del Problema Inverso

Para recuperar la geometría de referencia $\Omega_R$ a partir de la configuración observada $\Omega_{obs}$, se utiliza el método de **Movimiento Inverso** (Inverse Motion).

### Cinemática Inversa
Definimos el mapeo inverso $\boldsymbol{\chi}^{-1}: \Omega_{obs} \to \Omega_R$ mediante un campo de desplazamiento inverso $\mathbf{w}(\mathbf{x})$:
$$\mathbf{X} = \mathbf{x} - \mathbf{w}(\mathbf{x})$$

A diferencia de los métodos iterativos aditivos, aquí se formula el gradiente de deformación "forward" ($\mathbf{F}$) directamente en función de la incógnita $\mathbf{w}$:

1.  **Gradiente del mapeo inverso:** $\mathbf{f} = \nabla_{\mathbf{x}} \mathbf{X} = \mathbf{I} - \nabla_{\mathbf{x}} \mathbf{w}$
2.  **Gradiente de deformación constitutivo:** $\mathbf{F} = \mathbf{f}^{-1} = (\mathbf{I} - \nabla_{\mathbf{x}} \mathbf{w})^{-1}$

### Ecuación de Gobierno (Equilibrio)
El problema se resuelve imponiendo el equilibrio de fuerzas directamente sobre la malla deformada (Euleriana). Se busca el campo $\mathbf{w}$ que satisfaga la forma débil:

$$\int_{\Omega_{obs}} (\mathbf{F} \cdot \mathbf{S}(\mathbf{F})) : \nabla_{\mathbf{x}} \mathbf{v} \cdot \mathbf{F} \det(\mathbf{f}) \, d\Omega = 0 \quad \forall \mathbf{v}$$

Esta formulación permite encontrar la geometría stress-free en un **único paso de solución no lineal** (Newton-Raphson), evitando la necesidad de algoritmos de optimización costosos.

---

## 3. Hipótesis y Suposiciones del Modelo

La implementación actual realiza las siguientes asunciones para hacer el problema tratable computacionalmente:

1.  **Estado Cuasi-Estático:** Se asume que la configuración observada (CT) es un estado de equilibrio estático. Se desprecian los términos inerciales.
2.  **Surfactante "Congelado":** Para el paso inverso, se toma la distribución de surfactante observada ($\Gamma_{obs}$) como un dato de entrada fijo. No se resuelve la historia temporal de adsorción/desorción hacia atrás en el tiempo.
3.  **Presión Uniforme (Modelo de un Compartimento):** Se asume que la presión alveolar $P_{alv}$ es uniforme en todo el dominio (sin gradientes de presión por flujo de Darcy en el estado final).
4.  **Gravedad Despreciable:** Dado que las fuerzas elásticas y de tensión superficial dominan a escala alveolar, se desprecian las fuerzas másicas ($\mathbf{B} = \mathbf{0}$) en la formulación del equilibrio.
5.  **Homogeneidad Material:** Los parámetros constitutivos ($c, a, b, \Phi_R$) se asumen constantes espacialmente en esta versión del código.

---

## 4. Implementación Numérica

El problema se resuelve utilizando el **Método de Elementos Finitos (FEM)** con la librería [Firedrake](https://www.firedrakeproject.org/).

### Espacios Funcionales
Se utiliza una discretización mixta compatible:
* **Desplazamientos ($\mathbf{u}, \mathbf{w}$):** Elementos Lagrangianos Cuadráticos Continuos ($CG_2$ o $P_2$). Esto es necesario para capturar correctamente los gradientes de deformación finita.
* **Variables Internas (Surfactante $\Gamma$, Porosidad $\Phi$):** Elementos Discontinuos Constantes ($DG_0$ o $P_0$). Estas variables se definen localmente por elemento (*cell-wise*) y no requieren continuidad espacial.

### Solver No Lineal
* **Método:** Newton-Raphson con búsqueda de línea (Line Search).
* **Precondicionador:** Factorización LU directa (MUMPS) para robustez en 2D, o métodos iterativos para 3D.
* **Jacobiano:** Calculado mediante Diferenciación Automática Simbólica (UFL) proporcionada por Firedrake, asegurando la consistencia tangente exacta del modelo de surfactante altamente no lineal.

---

## 5. Parámetros del Modelo

Los siguientes parámetros, obtenidos de Avilés-Rojas & Hurtado (2025), se utilizan para caracterizar el tejido y el surfactante:

| Parámetro | Símbolo | Valor Típico | Descripción |
|-----------|:-------:|--------------|-------------|
| **Geometría** | | | |
| Porosidad Ref. | $\Phi_R$ | 0.74 | Fracción de aire en estado stress-free |
| Radio Alveolar | $R_{RI}$ | 0.11 mm | Escala de longitud microestructural |
| **Elasticidad** | | | |
| Rigidez | $c$ | 2.5 kPa | Módulo de rigidez del tejido |
| Exponente 1 | $a$ | 0.433 | No-linealidad isotrópica |
| Exponente 2 | $b$ | -0.610 | No-linealidad volumétrica |
| **Surfactante** | | | |
| Tensión Máx. | $\gamma_0$ | 70 mN/m | Tensión de agua pura / límite superior |
| Tensión Equil. | $\gamma_{inf}$| 22 mN/m | Tensión de equilibrio dinámico |
| Conc. Crítica | $\Gamma_{inf}$| 3.0$\times 10^{-9}$ g/mm² | Concentración de referencia |

## Referencias

1.  **Modelo Físico:** Avilés-Rojas, N., & Hurtado, D.E. (2025). Integrating pulmonary surfactant into lung mechanical simulations: A continuum approach to surface tension in poromechanics. *Journal of the Mechanics and Physics of Solids*, 203, 106174.
2.  **Método Numérico:** Govindjee, S., & Mihalic, P.A. (1996). Computational methods for inverse finite elastostatics. *Computer Methods in Applied Mechanics and Engineering*, 136, 47-57.