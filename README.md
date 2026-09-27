# Predicción de Velocidad de Adopción de Mascotas — Arquitectura Medallion

Pipeline de datos end-to-end con **PySpark + Delta Lake**, siguiendo arquitectura **Bronze → Silver → Gold**, para predecir qué tan rápido será adoptada una mascota (dataset [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/leaderboard), Kaggle).

Este repositorio implementa la parte de ingeniería de datos y modelado del paper **"Optimizando la Velocidad de Adopción de Mascotas"** (Universidad ESAN, 2025), que incluye revisión de estado del arte, metodología formal y análisis de resultados.

## Overview

El proyecto combina datos tabulares, texto (descripciones) e imágenes de las mascotas para predecir `AdoptionSpeed` (velocidad de adopción, variable categórica de 5 niveles). El dataset corresponde al registrado en 2019 en la competencia PetFinder de Kaggle, con 14,994 registros y 7 archivos fuente. El pipeline procesa estos datos a través de las tres capas de la arquitectura Medallion, con feature engineering extensivo y comparación de múltiples modelos de Machine Learning en Spark ML.

## Arquitectura

```
Bronze (raw)              Silver (limpio)            Gold (listo para ML)
─────────────             ──────────────              ──────────────────
train.csv                 Joins de referencias        Imputación de nulos
test.csv                  (raza, color, estado)        (media / ceros)
PetFinder-*Labels.csv  →  Parsing de vectores      →   Feature selection
features_imagenes.csv     (imagen, texto)              (Chi², ANOVA,
feature_texto.csv         Feature engineering           multicolinealidad)
                           (13+ features derivados)     Escritura en Delta
```

**Bronze**: ingesta de 7 fuentes (train/test de PetFinder, diccionarios de raza/color/estado, y features pre-extraídos de imágenes vía SIFT y de texto vía Word2Vec).

**Silver**: joins con tablas de referencia, parsing de vectores de imagen y texto (regex + funciones nativas de Spark SQL), imputación de categóricas, y feature engineering:

- `Img_Vector_Avg` — proxy de calidad/contraste visual de la foto
- `Desc_Magnitude` — "intensidad" semántica de la descripción (norma del vector W2V)
- `Rescuer_Count` — actividad histórica del rescatista (window function)
- `Has_Real_Name` — filtro de nombres genéricos ("no name", "kitten", etc.)
- `Health_Index`, `At_Risk`, `Maintenance_Level` — KPIs veterinarios derivados
- `Is_Urban_Center`, `Is_Solo`, `Color_Count`, `Is_Pure_Breed` — features de contexto

**Gold**: imputación final, deduplicación por `PetID`, y selección de features con una clase propia (`FeatureSelectorBDA`) que aplica:
- Filtro de multicolinealidad (correlación > 0.9) para numéricas
- Test ANOVA para relevancia de numéricas vs. target
- Chi² para relevancia de categóricas vs. target

Resultado: 19 features finales seleccionados de forma estadística, no manual.

## Resultados — Comparación de Modelos

Se entrenaron y compararon 5 algoritmos de clasificación multiclase en Spark ML, en dos escenarios:

**Caso A — dataset completo (5 clases, incluyendo clase 0 minoritaria):**

| Modelo | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.399 | 0.364 | 0.410 | 0.399 |
| MLP | 0.374 | 0.340 | 0.353 | 0.374 |
| Decision Tree | 0.369 | 0.347 | 0.372 | 0.369 |
| Logistic Regression | 0.341 | 0.295 | 0.324 | 0.341 |
| Naive Bayes | 0.319 | 0.284 | 0.354 | 0.319 |

**Caso B — sin clase 0 + oversampling + tuning (4 clases balanceadas):**

| Modelo | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|
| **Random Forest** | **0.423** | **0.402** | **0.409** | **0.423** |
| MLP | 0.378 | 0.362 | 0.366 | 0.378 |
| Decision Tree | 0.365 | 0.363 | 0.366 | 0.365 |
| Logistic Regression | 0.340 | 0.327 | 0.346 | 0.340 |
| Naive Bayes | 0.186 | 0.124 | 0.177 | 0.186 |

Random Forest fue el modelo ganador en ambos escenarios. El split train/test se hizo **antes** del oversampling (80/20) para evitar data leakage, y el oversampling se aplicó únicamente al set de entrenamiento.

**Limitaciones documentadas** (según el paper del proyecto): disponibilidad limitada de datos conductuales de las mascotas, variabilidad entre refugios no considerada en el dataset, y restricciones en el uso de modelos más complejos por recursos computacionales disponibles. El accuracy (~40-42%) refleja la dificultad real del problema — PetFinder es una competencia activa de Kaggle — más que una limitación del pipeline de datos, que fue el foco técnico principal del proyecto.

## Stack Técnico

- **Procesamiento distribuido**: PySpark, Spark SQL, Spark ML
- **Almacenamiento**: Delta Lake (formato ACID, versionado, time travel)
- **Feature engineering**: Window functions, UDFs, regex nativo de Spark SQL, vectores de imagen (OpenCV SIFT, 128 dim) y texto (Spark ML Word2Vec, 20 dim)
- **Selección de features**: Chi², ANOVA, análisis de multicolinealidad (clase propia `FeatureSelectorBDA`)
- **Modelos**: DecisionTree, RandomForest, MLP, NaiveBayes, LogisticRegression (Spark ML), con `CrossValidator` para tuning
- **Visualización**: Matplotlib, Seaborn (matrices de confusión)

## Dataset

[PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/leaderboard) (Kaggle, datos de 2019) — 14,994 perfiles de mascotas de Malasia, distribuidos en 7 archivos (train, test, diccionarios de raza/color/estado, y features pre-extraídos). Se complementó con:
- Features de imagen extraídos con SIFT (`FeatureExtraction-SIFT.ipynb`)
- Features de texto extraídos con Word2Vec (`FeatureExtraction-W2V.ipynb`)

## Estructura del repositorio

```
├── notebooks/
│   ├── FeatureExtraction-SIFT.ipynb      # Vectores de imagen (OpenCV SIFT, 128 dim)
│   ├── FeatureExtraction-W2V.ipynb       # Embeddings de texto (Spark ML Word2Vec, 20 dim)
│   ├── ClaseFeatureSelectorBDA.ipynb     # Clase de selección de features (Chi², ANOVA, multicolinealidad)
│   └── NotebookFinal.ipynb               # Pipeline Bronze → Silver → Gold → ML
├── data/                                  # (no versionado — ver sección Dataset)
│   ├── train/train.csv, test/test.csv
│   ├── train_images/, test_images/
│   └── BreedLabels.csv, ColorLabels.csv, StateLabels.csv
├── docs/
│   └── Optimizando_Velocidad_Adopcion.pdf # Paper del proyecto
└── README.md
```

## Cómo correrlo

```bash
pip install pyspark delta-spark scikit-learn matplotlib seaborn pandas opencv-python tqdm
```

1. Descarga el dataset de [Kaggle](https://www.kaggle.com/c/petfinder-adoption-prediction/leaderboard) y colócalo en `data/` siguiendo la estructura de arriba
2. Corre `FeatureExtraction-SIFT.ipynb` — procesa las imágenes en `train_images/` con SIFT (OpenCV) y genera `features_imagenes.csv` (toma ~30 min para las ~58,000 imágenes del dataset completo)
3. Corre `FeatureExtraction-W2V.ipynb` — entrena un modelo Word2Vec (Spark ML) sobre las descripciones y genera `feature_texto.csv`
4. Asegúrate que `ClaseFeatureSelectorBDA.ipynb` esté en la misma carpeta que `NotebookFinal.ipynb` (el pipeline lo importa con `%run`)
5. Corre `NotebookFinal.ipynb` de inicio a fin — genera automáticamente las capas Silver y Gold, entrena los modelos y guarda las predicciones finales en Delta

## Contexto académico

Desarrollado como parte del paper "Optimizando la Velocidad de Adopción de Mascotas" para el curso de Big Data en Universidad ESAN, Lima, Perú (2025). El paper incluye revisión de estado del arte (comparación con estudios similares en refugios de EE.UU., Indonesia y UK) y metodología completa.

## Autores

- Aaron Soto López
- Diego Alberto Chavez Polinar
- Gustavo Anderson Mendoza Montes
- Omar Chanca Huamán
- José Rojas Vargas
