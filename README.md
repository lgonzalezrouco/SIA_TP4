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
├── src/
│   ├── kohonen/        # Implementación de SOM
│   ├── oja/            # Implementación de la Regla de Oja
│   ├── hopfield/       # Implementación de la Red de Hopfield
│   └── utils/          # Carga de datos y visualización
├── config.json         # Hiperparámetros y configuración
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

El archivo `config.json` contiene los parámetros de los algoritmos:

```json
{
  "exercise": "kohonen",
  "kohonen": {
    "grid_size": 5,
    "epochs": 1000,
    "learning_rate": 0.1,
    "radius": 5
  },
  "oja": { ... },
  "hopfield": { ... }
}
```
