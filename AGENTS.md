# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project

TP4 for Sistemas de Inteligencia Artificial (ITBA, 2026). Implements three unsupervised learning algorithms applied to a European countries dataset and letter-pattern recognition:

1. **Kohonen SOM** — cluster 28 European countries by geopolitical/economic features
2. **Oja's Rule** — extract first principal component, compare with sklearn PCA
3. **Hopfield Network** — store/recall 5×5 letter patterns (±1), demonstrate spurious states

Dataset: `data/europe.csv` — 28 countries × 7 features (Area, GDP, Inflation, Life.expect, Military, Pop.growth, Unemployment).

## Commands

```bash
# Install dependencies
uv sync

# Add a dependency
uv add numpy matplotlib pandas scikit-learn seaborn

# Run the project
uv run python main.py --config config.json

# Run a specific exercise
uv run python main.py --config config.json --exercise kohonen

# Run tests (once test suite exists)
uv run pytest

# Run a single test
uv run pytest tests/test_kohonen.py::test_bmu -v
```

## Architecture

```text
src/
  kohonen/
    som.py        # Lógica del SOM (entrenamiento, BMU, pesos)
    experiment.py # Orquestador (flujo train -> analyze)
    plots.py      # Visualizaciones específicas (U-Matrix, Heatmaps)
  oja/
    oja_neuron.py # Neurona lineal y Regla de Oja
    experiment.py # Entrenamiento y comparación con PCA
    plots.py      # Gráficos de convergencia
  hopfield/
    hopfield_net.py # Lógica de almacenamiento y recuperación
    patterns.py     # Definición de letras y funciones de ruido
    experiment.py   # Pruebas de memoria asociativa
  utils/
    data_loader.py   # Carga y normalización Z-score
    config_parser.py # Validación y parseo del config.json
data/
  europe.csv
results/          # Directorio para guardar plots y reportes
config.json       # Hiperparámetros
main.py           # CLI entry point - orquestador principal
```

`main.py` reads a JSON/YAML config file and an optional `--exercise` flag to run one or all exercises.

## Algorithm Implementation Notes

### Kohonen SOM

- **Preprocessing**: z-score normalize all 7 features before training. The raw features have very different scales (Area in km² vs. Inflation %).
- **Performance**: use fully vectorized numpy — compute all BMU distances as a matrix operation (`np.linalg.norm(weights - sample, axis=2)`), never loop over neurons.
- **Neighborhood function**: Gaussian decay over grid distance; both learning rate and radius decay exponentially with epoch.
- **Required outputs**: country-to-neuron assignment heatmap, U-matrix (average distance to neighbors), neuron activation count per country.

### Oja's Rule

- **Model**: single linear neuron, weight vector same dimension as input (7).
- **Update**: `w += lr * (x * y - y**2 * w)` where `y = w · x` (scalar).
- **Convergence check**: stop when `||Δw|| < tol` across a full epoch.
- **Comparison**: after convergence, compare `w` (normalized) against `sklearn.decomposition.PCA(n_components=1).components_[0]` — signs may differ, check alignment by dot product.

### Hopfield Network

- **Storage**: weight matrix `W = (1/N) * Σ ξ_i ξ_i^T`, zero diagonal.
- **Retrieval**: asynchronous update (random unit order each step) until convergence or max steps.
- **Spurious states**: demonstrate by initializing with a very noisy pattern (>50% bits flipped) that converges to a mixed/spurious attractor, not any stored pattern.
- **Patterns**: 5×5 matrices flattened to 25-dim vectors of ±1. Store 4 distinct letters.

## Performance

- Vectorize all inner loops with numpy — especially the Kohonen BMU search and weight update.
- For Hopfield, the weight matrix is 25×25; full matrix operations are fine.
- Kohonen grids up to ~10×10 and ≤1000 epochs should run in seconds with vectorized code.
- Profile with `uv run python -m cProfile -s cumtime main.py --config config.json` if a run exceeds ~30s.
