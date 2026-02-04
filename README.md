# 🌱 Sistema de recomendación de cultivos basado en Machine Learning 🌱

Este proyecto desarrolla un **sistema de recomendación de cultivos** utilizando técnicas de **machine learning supervisado**, a partir de datos del suelo y variables ambientales.

El objetivo es **predecir el cultivo más adecuado** dadas condiciones específicas de:
- nutrientes del suelo
- clima
- precipitaciones

Este enfoque permite apoyar la **toma de decisiones agrícolas basada en datos**, mejorando productividad, sostenibilidad y uso eficiente de recursos.

---

## 🌱 Problema a resolver

Elegir el cultivo adecuado es una decisión crítica en agricultura.  
Factores como la composición del suelo (N, P, K), el pH y las condiciones climáticas influyen directamente en el rendimiento.

Este proyecto aborda el problema como una **tarea de clasificación multiclase**, donde:
- **Input:** condiciones del suelo y ambientales
- **Output:** tipo de cultivo recomendado

---

## 🎯 Objetivo de Machine Learning

- **Tipo de problema:** Clasificación multiclase
- **Variable objetivo:** `Crop`
- **Enfoque:** comparar múltiples algoritmos y seleccionar el modelo con mejor desempeño general

---

## 📊 Dataset

El dataset incluye las siguientes variables:

- **Nutrientes del suelo**
  - Nitrogen
  - Phosphorus
  - Potassium
- **Variables ambientales**
  - Temperature
  - Humidity
  - pH_Value
  - Rainfall
- **Target**
  - Crop (tipo de cultivo)

El conjunto de datos se encuentra **balanceado**, lo que reduce el riesgo de sesgo en los modelos de clasificación.

---

## 🧪 Metodología

1. **Análisis exploratorio de datos (EDA)**
   - Distribuciones
   - Detección de outliers
   - Análisis de correlaciones
2. **Feature analysis**
   - Evaluación de colinealidad (P vs K)
   - Importancia de características usando ExtraTrees
3. **Preparación para ML**
   - Separación train/test
   - Codificación de la variable objetivo
4. **Modelado**
   - Entrenamiento y comparación de múltiples clasificadores
5. **Optimización**
   - Búsqueda de hiperparámetros con GridSearchCV
6. **Evaluación**
   - Accuracy
   - Precision, Recall y F1-score
   - Matriz de confusión

---

## 🤖 Modelos evaluados

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Extra Trees Classifier
- Naive Bayes (Gaussian y Multinomial)
- Quadratic Discriminant Analysis
- Bagging Classifier
- LightGBM Classifier

---

## 🏆 Modelo final seleccionado

**ExtraTreesClassifier**  
Parámetros óptimos:
- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`

Este modelo presentó:
- Alto rendimiento en validación cruzada
- Buen balance entre sesgo y varianza
- Métricas sólidas en el conjunto de prueba

---

## 📈 Métricas de evaluación

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score
- Matriz de confusión

El desempeño consistente en todas las clases indica un **modelo robusto para recomendación de cultivos**.

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **pandas, numpy**
- **matplotlib, seaborn**
- **scikit-learn**
- **LightGBM**
- **SciPy**

---

## 📂 Estructura del repositorio

├── Crop_Recommendation.csv
├── Recomendación de cultivos.py
├── README.md


---

## 🚀 Próximos pasos

- Implementar un pipeline completo con `sklearn.pipeline`
- Evaluar técnicas de normalización y escalado
- Incorporar explainability (SHAP / feature importance local)
- Desplegar el modelo como API (FastAPI / Flask)
- Integrar datos geográficos o temporales

---

## 👤 Autor

**Flavia Hepp**  
Data Scientist en formación  
