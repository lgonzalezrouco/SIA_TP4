# SIA TP4 - Aprendizaje No Supervisado

Este proyecto implementa tres algoritmos de aprendizaje no supervisado como parte del Trabajo Práctico 4 de Sistemas de Inteligencia Artificial (ITBA, 2026).

## Algoritmos Implementados

1. **Kohonen SOM (Mapas Auto-Organizados)**: Clasificación de 28 países europeos basados en variables geopolíticas y económicas.
2. **Regla de Oja**: Extracción del primer componente principal (PCA) mediante una neurona lineal.
3. **Red de Hopfield**: Memoria asociativa para almacenar y recuperar patrones de letras en una grilla de 5×5.

## Estructura del Proyecto

```text
SIA_TP4/
├── data/               # Datos de entrada (europe.csv)
├── results/            # Gráficos y resultados exportados
├── src/
│   ├── kohonen/        # Implementación de SOM (som.py, experiment.py, plots.py)
│   ├── oja/            # Implementación de Oja (oja_neuron.py, experiment.py, plots.py)
│   ├── hopfield/       # Implementación de Hopfield (hopfield_net.py, experiment.py, patterns.py)
│   └── utils/          # Carga de datos y validación de config (data_loader.py, config_parser.py)
├── config.example.json # Plantilla de configuración (versionada)
├── config.json         # Hiperparámetros locales (ignorado por git)
├── main.py             # Punto de entrada de la CLI
└── pyproject.toml      # Dependencias y configuración del proyecto
```

## Requisitos

Este proyecto utiliza `uv` para la gestión de dependencias y entornos virtuales.

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Instalación

Para instalar las dependencias y configurar el entorno:

```bash
uv sync
```

Antes de la primera ejecución, copiá la plantilla:

```bash
cp config.example.json config.json
```

En PowerShell:

```powershell
Copy-Item config.example.json config.json
```

## Ejecución

El script principal `main.py` permite ejecutar los diferentes ejercicios utilizando un archivo de configuración JSON.

```bash
# Ejecutar el ejercicio por defecto (definido en config.json)
uv run main.py

# Ejecutar un ejercicio específico
uv run main.py --exercise kohonen
uv run main.py --exercise oja
uv run main.py --exercise hopfield

# Usar un archivo de configuración personalizado
uv run main.py --config mi_configuracion.json
```

## Configuración

El archivo `config.json` contiene los parámetros de los algoritmos. La
plantilla en `config.example.json` muestra la estructura completa para
Kohonen:

```json
{
  "exercise": "kohonen",
  "kohonen": {
    "grid_rows": 5,
    "grid_cols": 5,
    "topology": "rectangular",
    "init_method": "sample",
    "distance": "euclidean",
    "neighborhood": "gaussian",
    "learning_rate": {"kind": "exponential", "initial": 0.5, "final": 0.01},
    "radius": {"kind": "exponential", "initial": 3.0, "final": 1.0},
    "epochs": 500,
    "seed": 42,
    "grid_sizes_sweep": ["3x3", "4x4", "5x5", "6x6", "7x7", "8x8"]
  },
  "oja": { ... },
  "hopfield": { ... }
}
```

`learning_rate` y `radius` aceptan tanto un número (constante) como un
diccionario con `kind` ∈ `{constant, linear, exponential, inverse}` para
schedules variables.

`grid_sizes_sweep` lista tamaños como `"5x5"` o `"10x8"` para el barrido de
grilla: genera `10_sweep_grid_size.png`, un `10_country_map_{RxC}.png` por
cada tamaño (misma vista que `01_country_map.png`) y opcionalmente
`grid_sweep_seeds` (default `5`) para el promedio de error de cuantización.

## Ejercicio 2 - Red de Hopfield

El alfabeto se carga desde `data/letters.json` (26 letras de 5×5 codificadas
con 0/1 y mapeadas a ±1). El experimento se controla con la sección
`hopfield` del `config.json` y se ejecuta con:

```bash
uv run main.py --exercise hopfield
```

### Parámetros principales

- `shape`: dimensiones de los patrones (default `[5, 5]`).
- `letters_path`: archivo de letras (default `data/letters.json`).
- `patterns`: letras que se almacenan en la red (default `["A","B","C","D"]`).
- `combo_k`: tamaño de las combinaciones para el análisis de ortogonalidad
  (default igual a `len(patterns)`).
- `noise.n_flips` y `noise.trials`: ruido (bits invertidos) y repeticiones
  para el experimento de recall.
- `spurious.n_flips` y `spurious.trials`: ruido alto y repeticiones para
  detectar estados espurios.
- `capacity.{p_values, noise_flips, trials}`: barridos para el heatmap de
  capacidad (activable con `analyses.capacity = true`).
- `analyses`: flags para encender/apagar cada análisis (`orthogonality`,
  `recall`, `spurious`, `single_pixel`, `capacity`, `complete_sweep`).
- `sweep`: barrido dimensión × p — ortogonalidad, elige mejor/peor grupo y
  recall ruidoso. Ver `sweep.dimensions`, `sweep.p_values`, `sweep.compare_fixed`.

### Barrido completo (`complete_sweep`)

Con `analyses.complete_sweep: true` se ejecuta para cada dimensión en
`sweep.dimensions` y cada `p` en `sweep.p_values`:

1. C(n, p) combinaciones → PI medio / máx → CSV y scatter.
2. Entrena y hace recall con el **mejor** y el **peor** grupo (por PI medio).
3. Si `len(compare_fixed) == p`, también evalúa ese grupo fijo (ej. `ALTV`).

Salida en `results/hopfield/{5x5,7x7}/p{p}/best|worst|fixed/...` y
`sweep_summary.csv`.

### Outputs en `results/hopfield/`

- `letters_grid.png` — todas las letras del alfabeto.
- `dot_product_matrix.png` — heatmap del producto interno entre todas las
  letras.
- `combinations_all.csv`, `combinations_top_avg_low.csv`,
  `combinations_top_avg_high.csv`, `combinations_top_max_low.csv` —
  análisis sobre las C(n, k) combinaciones (PI medio, PI máx, count).
- `combinations_scatter.png` — PI medio vs PI máx para cada grupo.
- `group_dot_matrix.png` — matriz de PI del grupo elegido.
- `recall/{letra}_steps.png`, `recall/{letra}_energy.png`, `recall/summary.csv`
  — recuperación paso a paso y curva de energía.
- `spurious/examples.png`, `spurious/summary.csv`, `spurious/per_source.csv`
  — estados espurios encontrados.
- `single_pixel/results.csv` — robustez ante 1 píxel invertido.
- `capacity_heatmap.png`, `capacity.csv` — tasa de recall vs `p` y ruido
  (sólo si `analyses.capacity = true`).
