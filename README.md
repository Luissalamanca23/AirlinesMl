# Airlines ML - Predicción de Precios de Vuelos

Un proyecto de machine learning para predecir precios de boletos de aerolíneas utilizando el framework Kedro.

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de machine learning para predecir precios de vuelos basado en varias características de aerolíneas y vuelos. El pipeline está construido usando Kedro, asegurando flujos de trabajo de ML reproducibles y mantenibles.

## Dataset

El dataset contiene 300,153 registros de vuelos con las siguientes características:

- **Airline**: Nombre de la compañía aérea (6 categorías)
- **Flight**: Código identificador del vuelo
- **Source City**: Ciudad de origen (6 ciudades únicas)
- **Departure Time**: Períodos de tiempo de salida categorizados (6 etiquetas)
- **Stops**: Número de escalas (zero, one, two_or_more)
- **Arrival Time**: Períodos de tiempo de llegada categorizados (6 etiquetas)
- **Destination City**: Ciudad de destino (6 ciudades únicas)
- **Class**: Clase del asiento (Business, Economy)
- **Duration**: Duración del vuelo en horas (continuo)
- **Days Left**: Días restantes hasta la salida del vuelo
- **Price**: Precio del boleto (variable objetivo)

## Estructura del Proyecto

```
AirlinesMl/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Configuración del catálogo de datos
│   │   └── parameters.yml       # Parámetros del modelo
│   └── local/
│       └── credentials.yml      # Credenciales locales
├── data/
│   ├── 01_raw/                  # Datos crudos
│   ├── 02_intermediate/         # Datos procesados
│   ├── 03_primary/              # Datos limpios
│   ├── 04_feature/              # Datos con ingeniería de características
│   ├── 05_model_input/          # Datos listos para el modelo
│   ├── 06_models/               # Modelos entrenados
│   ├── 07_model_output/         # Predicciones del modelo
│   └── 08_reporting/            # Reportes y métricas
├── src/airlines_ml/
│   └── pipelines/
│       ├── data_processing/     # Pipeline de procesamiento de datos
│       └── data_science/        # Pipeline de ML
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Pruebas unitarias
└── requirements.txt             # Dependencias del proyecto
```

## Instalación

1. Clona el repositorio
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Instala el proyecto en modo desarrollo:
   ```bash
   pip install -e .
   ```

## Uso

### Ejecutar el pipeline completo
```bash
kedro run
```

### Ejecutar pipelines específicos
```bash
kedro run --pipeline data_processing
kedro run --pipeline data_science
```

### Visualizar el pipeline
```bash
kedro viz
```

### Lanzar Jupyter notebooks
```bash
kedro jupyter notebook
```

## Pipeline de Procesamiento de Datos

El pipeline de procesamiento de datos incluye:

1. **Limpieza de Datos**: Manejar valores faltantes y remover duplicados
2. **Preprocesamiento de Datos**: Convertir variables categóricas e ingeniería básica de características
3. **División de Datos**: Dividir datos en conjuntos de entrenamiento (64%), validación (16%) y prueba (20%)

## Pipeline de Machine Learning

El pipeline de ciencia de datos implementa múltiples algoritmos:

- Regresión Lineal (modelo base)
- Random Forest Regressor
- XGBoost
- LightGBM

Los modelos son evaluados usando:
- Error Cuadrático Medio (RMSE)
- Error Absoluto Medio (MAE)
- R-cuadrado (R²)

## Resultados del Modelo

1. **Random Forest**: RMSE: 2,623.38, MAE: 1,257.08, R²: 0.9866 ⭐ **Mejor modelo**
2. **XGBoost**: RMSE: 3,441.46, MAE: 1,924.01, R²: 0.9770
3. **LightGBM**: RMSE: 3,589.68, MAE: 2,047.23, R²: 0.9750
4. **Regresión Lineal**: RMSE: 7,012.90, MAE: 4,626.32, R²: 0.9046

## Desarrollo

### Ejecutar pruebas
```bash
pytest
```

### Verificaciones de calidad de código
```bash
kedro lint
```

## Configuración

- **Catálogo de Datos**: Configurar datasets en `conf/base/catalog.yml`
- **Parámetros**: Establecer parámetros del modelo en `conf/base/parameters.yml`
- **Credenciales**: Almacenar información sensible en `conf/local/credentials.yml`

## Tecnologías

- **Kedro**: Framework de pipelines
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Algoritmos de machine learning
- **XGBoost**: Gradient boosting
- **LightGBM**: Gradient boosting rápido
- **Kedro-Viz**: Visualización de pipelines

## Autores

- Luis Salamanca
- Brahian Gonzales

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.