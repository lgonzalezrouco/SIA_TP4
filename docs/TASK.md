# Aprendizaje No Supervisado

## Trabajo Práctico 4

### Sistemas de Inteligencia Artificial 2026

## 1. Ejercicio Europa

El conjunto de datos `europe.csv` corresponde a características económicas, sociales y geográficas de 28 países de Europa. Las variables son:

- **Country**: Nombre del país.
- **Area**: Área.
- **GDP**: Producto Bruto Interno.
- **Inflation**: Inflación anual.
- **Life.expect**: Expectativa de vida media en años.
- **Military**: Presupuesto militar.
- **Pop.growth**: Tasa de crecimiento poblacional.
- **Unemployment**: Tasa de desempleo.

### 1.1. Red de Kohonen

Implementar la red de Kohonen y aplicarla para resolver los siguientes problemas:

- Asociar países que posean las mismas características geopolíticas, económicas y sociales.
- Realizar al menos un gráfico que muestre los resultados.
- Realizar un gráfico que muestre las distancias promedio entre neuronas vecinas.
- Analizar la cantidad de elementos que fueron asociados a cada neurona.

### 1.2. Modelo de Oja

Implementar una red neuronal utilizando la regla de Oja para resolver los siguientes problemas:

- Calcular la primera componente principal para este conjunto de datos.
- Interpretar el resultado de la primera componente.
- Comparar el resultado del ejercicio de Oja con el resultado de calcular la primera componente principal con una librería.

## 2. Ejercicio Patrones

### 2.1. Modelo de Hopfield

Construir patrones de letras del abecedario utilizando 1 y -1 y matrices de 5x5. Por ejemplo, con la matriz:

```text
 1  1  1  1  1
-1 -1 -1  1 -1
-1 -1 -1  1 -1
 1 -1 -1  1 -1
 1  1  1 -1 -1
```

puede dibujarse la letra J:

```text
* * * * *
      *
      *
*     *
* * *
```

- Almacenar 4 patrones de letras. Implementar el modelo de Hopfield para asociar matrices ruidosas de 5x5 con los patrones de las letras almacenadas. Los patrones de consulta deben ser alteraciones aleatorias de los patrones originales. Mostrar los resultados que se obtienen en cada paso hasta llegar al resultado final.
- Ingresar un patrón muy ruidoso e identificar un estado espúreo.

## 3. Entrega

Entregar por Campus la presentación, el repositorio con `README.md` y archivo de configuración y el hash del commit.
