# 🏠 Proyecto 7: House Rental Price Predictor

![House rental price predictor](https://raw.githubusercontent.com/SrAlcast/Proyecto7-PrediccionCasas/refs/heads/main/src/House%20rental%20price%20predictor.jpg)

## 📖 Descripción del Proyecto

Este proyecto tiene como objetivo principal desarrollar un modelo predictivo para estimar el precio de propiedades en Madrid
utilizando datos reales. El análisis se enfoca en comprender las principales variables que influyen en los precios y construir
un modelo interactivo mediante Streamlit para realizar predicciones basadas en las características de cada propiedad.

El mercado inmobiliario es complejo, dinámico y está influenciado por múltiples factores, como la ubicación, tamaño, y 
disponibilidad de servicios. Este proyecto busca aportar una herramienta útil tanto para compradores como vendedores.

## 🗂️ Estructura del Proyecto

El proyecto sigue la siguiente estructura de carpetas:

```
├── data/                # Contiene los datos originales
├── models/              # Modelos entrenados
├── notebooks/           # Notebooks para análisis y modelado
│   ├── 1-Preprocesing/  # Preprocesamiento de los datos
│   └── 2-Modeling/      # Entrenamiento y evaluación de modelos
├── src/                 # Scripts para tareas específicas
├── streamlit.py         # Aplicación interactiva en Streamlit
├── README.md            # Descripción general del proyecto
└── .gitattributes       # Configuración del repositorio
```

## 🛠️ Instalación y Requisitos

Este proyecto utiliza **Python 3.8** y requiere las siguientes bibliotecas:

- [Pandas](https://pandas.pydata.org/docs/) para manipulación de datos.
- [NumPy](https://numpy.org/doc/) para cálculos numéricos.
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html) para entrenamiento y evaluación de modelos.
- [Matplotlib](https://matplotlib.org/stable/contents.html) y [Seaborn](https://seaborn.pydata.org/) para visualización de datos.
- [Streamlit](https://docs.streamlit.io/) para la construcción de aplicaciones interactivas.

### Instrucciones de Instalación:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/SrAlcast/Proyecto7-PrediccionCasas.git
   ```

2. Instala las dependencias:
   ```bash
   pip install (biblioteca)
   ```

## 📋 Pasos para el Desarrollo

1. **Exploración de Datos**:
   - Analizar las principales características del dataset, identificar patrones y posibles valores atípicos.
   - Variables clave incluyen: `price`, `size`, `rooms`, `bathrooms`, `district`, entre otras.

2. **Preprocesamiento**:
   - Limpieza del dataset (manejo de valores nulos, duplicados).
   - Codificación de variables categóricas y escalado de variables numéricas.

3. **Entrenamiento del Modelo**:
   - Entrenar modelos como Regresión Lineal, Random Forest o Gradient Boosting.
   - Comparar métricas como RMSE y R² para seleccionar el mejor modelo.

## 📊 Resultados y Conclusiones

- Se han construido 4 modelos predictivos capaces de estimar los precios con base en las características principales.
- Estos modelos han sido:
#### **1. Linear Regression**
Modelo simple que asume una relación lineal entre las variables. Es rápido y fácil de interpretar, pero no captura relaciones no lineales.

#### **2. Decision Tree Regressor**
Divide los datos en regiones basadas en reglas simples. Captura relaciones no lineales, pero es propenso al sobreajuste sin regular.

#### **3. Random Forest Regressor**
Ensamble de múltiples árboles de decisión que reduce la varianza y generaliza bien. Es más preciso que un solo árbol, pero menos interpretable.

#### **4. Gradient Boosting Regressor**
Modelo secuencial que corrige errores iterativamente. Muy preciso en problemas complejos, aunque más lento y sensible al sobreajuste.

- El modelo que mejores metricas a proporcionado sorprendentemente ha sido Linear Regression

## 🤝 Contribuciones

Si deseas contribuir, puedes hacerlo a través de:
- **Pull Requests:** Comparte tus mejoras o nuevas funcionalidades.
- **Issues:** Reporta problemas o sugiere ideas. 
