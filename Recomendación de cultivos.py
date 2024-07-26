"""RECOMENDACIÓN DE CULTIVO

INTRODUCCIÓN

En agricultura, la recomendación precisa de cultivos es fundamental para garantizar un rendimiento óptimo y la sostenibilidad. A 
medida que los agricultores y expertos agrícolas profundizan en enfoques basados ​​en datos, se vuelve cada vez más evidente la 
importancia de aprovechar conjuntos de datos completos, en particular aquellos sobre la composición del suelo. El conjunto de datos 
considerado incorpora una gran cantidad de información que abarca factores clave como los niveles de nitrógeno, fósforo y potasio, 
junto con variables ambientales como la temperatura, la humedad, el valor del pH y las precipitaciones. Comprender y analizar este 
conjunto de datos es fundamental para tomar decisiones informadas que puedan mejorar la productividad agrícola, la gestión de recursos 
y la salud general de los cultivos.

¿POR QUÉ ES NECESARIO RECOMENDAR CULTIVOS?

1. Rendimiento óptimo:
las recomendaciones de cultivos personalizadas basadas en datos del suelo maximizan el potencial de rendimiento al abordar las 
deficiencias de nutrientes y optimizar las condiciones ambientales para el crecimiento de los cultivos.

2. Gestión de recursos:
las recomendaciones basadas en datos mejoran la asignación de recursos al garantizar el uso eficiente de fertilizantes, agua y otros 
insumos, lo que conduce a ahorros de costos y reducción del impacto ambiental.

3. Sostenibilidad:
Las recomendaciones precisas sobre cultivos contribuyen a las prácticas agrícolas sostenibles al promover la salud del suelo, minimizar 
la escorrentía de nutrientes y reducir la dependencia de insumos sintéticos.

4. Resiliencia climática:
las recomendaciones personalizadas mejoran la resiliencia de los cultivos a las variaciones climáticas, ayudando a los agricultores a 
adaptarse a las condiciones ambientales cambiantes y mitigar los riesgos asociados con eventos climáticos extremos.

5. Mayor rentabilidad:
las opciones de cultivos optimizadas basadas en datos del suelo aumentan las ganancias de los agricultores a través de mayores 
rendimientos, mejor calidad de los cultivos y menores costos de insumos.

CONCLUSIÓN

No se puede subestimar la importancia de aprovechar los datos del suelo para las recomendaciones de cultivos en la agricultura moderna. 
Al aprovechar los conocimientos proporcionados por conjuntos de datos integrales que abarcan la composición del suelo y las variables 
ambientales, los agricultores y expertos agrícolas pueden tomar decisiones informadas que optimicen la productividad de los cultivos, 
mejoren las prácticas de gestión de recursos, promuevan la sostenibilidad, refuercen la resiliencia climática y, en última instancia, 
mejoren la rentabilidad del sector agrícola. La toma de decisiones basada en datos es la piedra angular de estrategias eficaces de 
recomendación de cultivos, allanando el camino para un futuro agrícola más eficiente, resiliente y sostenible."""


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats # funciones estadísticas
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression  # Importar regresión lineal y regresión logística
from sklearn.ensemble import RandomForestRegressor  # Importar RandomForestRegressor para bosque aleatorio
from sklearn.svm import SVR  # Importar SVR para regresión de vectores de soporte
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Importar DecisionTree para árbol de decisión
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier # implementación del algoritmo K-Vecinos más cercanos para predicción continua de valores
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB # Importar naive_bayes
from sklearn.model_selection import train_test_split # dividir datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score # evaluar la precisión del clasificador
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')  # Ignorar mensajes de advertencia#Limpieza de datos
#Seleccionar el modelo de predicción correcto
#from lazypredict.Supervised import LazyClassifier
#A través del proceso integral de búsqueda de cuadrícula, hemos identificado el Clasificador de árboles adicionales como el modelo 
# de mejor rendimiento con los siguientes parámetros óptimos: {'max_ Depth': Ninguno, 'min_samples_split': 2, 'n_estimators': 200}. 
# Esta selección sienta las bases para construir un modelo de predicción altamente efectivo, adaptado a nuestro conjunto de datos y 
# objetivos específicos.
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
#Ajuste del modelo
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier


#CargadeDatos
df= pd.read_csv('Crop_Recommendation.csv')
plt.style.use('dark_background')
df.head() #para mostrar los destellos del conjunto de datos

print(f'Número de valores nulos: \n{df.isnull().any()}')
print(f'Número de valores duplicados: {df.duplicated().any()}')

#No hay valores nulos ni duplicados en el conjunto de datos anterior
df.nunique()

describe= df.describe().T
print(describe)

#Análisis univariado
# Seleccione columnas específicas de su DataFrame
selected_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# Determinar el número de filas y columnas para la cuadrícula de la trama secundaria
num_rows = (len(selected_columns) + 1) // 2  # Redondear al entero más cercano
num_cols = 2

plt.figure(figsize=(15, 15))
for i, col in enumerate(selected_columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.histplot(data=df, x=col, kde=True, bins=round(np.sqrt(len(df))), color='b')

plt.tight_layout()  # Ajustar el espacio entre subtramas
plt.show()

#Aquí la mayoría de las características no siguen la distribución normal. Algunas de las características son correctas y otras están 
# sesgadas a la izquierda.

#para mostrar las correlaciones entre diferentes características
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap = "inferno")
plt.title("Correlaciones entre diferentes características\n", fontsize = '16', fontweight = 'bold')
plt.show()

#El potasio y el fósforo están altamente correlacionados entre sí. Esta fuerte correlación puede conducir a colinealidad, lo que puede 
# afectar negativamente las producciones predictivas.

#Selecciones de funciones
X= df.drop("Crop",axis= 1) #asume que X es una característica independiente

y= df['Crop'] #supongamos que y es una variable objetivo

feat_selection = ExtraTreesClassifier()
feat_selection.fit(X, y)

feat_importances = pd.Series(feat_selection.feature_importances_, index=X.columns)  # Nombre de variable corregido
feat_importances.nlargest(len(df.columns)).plot(kind='barh')
plt.ylabel("Características\n")
plt.xlabel("Importancia de las características\n")
plt.title("Importancias de características (clasificador de árboles adicionales)\n", fontsize = '16', fontweight = 'bold')
plt.show()

#Queremos prever algunas precisiones de entrenamiento y prueba de diferentes algoritmos específicamente
#Aplicar la clasificación KNN
# Divida correctamente los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

train_score = {}
test_score = {}
n_neighbors = np.arange(2, 30, 1)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_score[neighbor]=knn.score(X_train, y_train)
    test_score[neighbor]=knn.score(X_test, y_test)

print(f'Precisiones del tren: \n{train_score}\n\nExactitudes de la prueba:\n{test_score}')

plt.plot(n_neighbors, train_score.values(), label="Precisión del tren", color = "fuchsia", marker = "v")
plt.plot(n_neighbors, test_score.values(), label="Precisión de la prueba", color = "limegreen", marker = "v")
plt.xlabel("Numero de vecinos\n")
plt.ylabel("Exactitud\n")
plt.title("KNN: número variable de vecinos\n", fontsize = '16', fontweight = 'bold')
plt.legend()
plt.grid()
plt.show()

#Crear un clasificador de KNeighbors
model = KNeighborsClassifier(n_neighbors = 20)

# Ajustar el clasificador a los datos de entrenamiento.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f'Precisión del tren: {train_accuracy}')
print(f'Precisión de la prueba: {test_accuracy}')

#Aplicación del clasificador Gaussiano Naive Bayes
# Crear un clasificador Gaussiano Naive Bayes
GNB = GaussianNB()

# Ajustar el clasificador a los datos de entrenamiento.
GNB.fit(X_train, y_train)
y_pred_gnb = GNB.predict(X_test)

print(f'Precisión del tren para GNB: {GNB.score(X_train, y_train)}')
print(f'Precisión de la prueba para GNB: {GNB.score(X_test, y_test)}')

#Aplicando clasificador MultinomialNB
## Crear un clasificador MultinomialNB
MNB = MultinomialNB()

MNB.fit(X_train, y_train)
y_pred_mnb = MNB.predict(X_test)

#exactitud
print(f'Precisión del tren para GNB: {MNB.score(X_train, y_train)}')
print(f'Precisión de la prueba para GNB: {MNB.score(X_test, y_test)}')

#Aplicar el modelo de árbol de decisión
# Inicializar y entrenar el modelo de árbol de decisión
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Calcule la precisión del entrenamiento y las pruebas.
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("Precisión del entrenamiento: ", train_accuracy)
print("Precisión de las pruebas: ", test_accuracy)

#Dado que el modelo no está bien ajustado, tenemos que ajustar el mejor parámetro
# Definir la grilla de parámetros
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 50, 20, 15],
    'min_samples_split': [2, 7, 10],
    'min_samples_leaf': [1, 2, 3]
}

# Inicializar el modelo
model = DecisionTreeClassifier()

# Inicializar GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Ajustar GridSearchCV a los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtenga los mejores parámetros y estimador
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print('Mejores hiperparámetros:', best_params)

# Inicializar el modelo con hiperparámetros especificados
model = DecisionTreeClassifier(criterion='entropy', max_depth=50, min_samples_leaf=1, min_samples_split=2)

# Ajustar el modelo a los datos de entrenamiento.
model.fit(X_train, y_train)
y_pred_dc= model.predict(X_test)

# Calcule e imprima la precisión del entrenamiento y las pruebas.
print("Precisión del entrenamiento:", model.score(X_train, y_train))
print("Precisión de las pruebas:", model.score(X_test, y_test))

#trazar el árbol de decisiones
plt.figure(figsize=(15,15))
plot_tree(model, filled=True, feature_names=X.columns, class_names= model.classes_)
plt.show()

#Clasificación forestal aleatoria
rf= RandomForestClassifier()

# Ajustar el modelo a los datos de entrenamiento.
rf.fit(X_train,y_train)

y_pred_rf= rf.predict(X_test)

# Calcule e imprima la precisión del entrenamiento y las pruebas.
print("Precisión del entrenamiento:", model.score(X_train, y_train))
print("Precisión de las pruebas:", model.score(X_test, y_test))

#Comparando diferentes modelos
# Inicializar los modelos
models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=50, min_samples_leaf=1, min_samples_split=2),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=20),
    'Gaussian Nayeb' : GaussianNB(),
    'MultinomialNb': MultinomialNB()
}

#Comparando diferentes modelos
# Inicializar los modelos
accuracies = {'Model': [], 'Training Accuracy': [], 'Testing Accuracy': []}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    accuracies['Model'].append(model_name)
    accuracies['Training Accuracy'].append(train_acc)
    accuracies['Testing Accuracy'].append(test_acc)
    print(f'{model_name} -Training Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}')

# Convierta el diccionario de precisiones a DataFrame para una mejor visualización
acc_df = pd.DataFrame(accuracies)

# Trazar las precisiones
plt.figure(figsize=(15, 15))
plt.bar(acc_df['Model'], acc_df['Training Accuracy'], alpha=0.6, color='gold', label='Precisión del entrenamiento\n')
plt.bar(acc_df['Model'], acc_df['Testing Accuracy'], alpha=0.6, color='mediumorchid', label='Precisión de las pruebas\n')
plt.xlabel('Nombres de modelos\n')
plt.ylabel('Exactitudes\n')
plt.title('Comparación de modelos: precisión del entrenamiento frente a las pruebas\n', fontsize = '16', fontweight = 'bold')
plt.legend()
plt.show()

#Perspectivas generales:
#El potasio y el fósforo están altamente correlacionados entre sí. por lo que la producción puede verse afectada con estas características.
#El clasificador de árbol de decisión y el clasificador de bosque aleatorio obtienen una gran precisión de entrenamiento y prueba para 
# predecir cultivos con nuevos datos.


df['Crop'] = pd.factorize(df['Crop'])[0] + 1
df.head()

#Con nuestros datos meticulosamente limpios, estamos listos para embarcarnos en el viaje revelador del análisis exploratorio de datos 
# (EDA) para descubrir patrones e historias ocultos en su interior.
#Análisis de datos exploratorios
features = list(df.drop(columns='Crop').columns)
data = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt='.2f', cmap='inferno', linewidths=0.5)

plt.title('Mapa de calor de correlación\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Características\n')
plt.ylabel('Características\n')

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Crop', data=df)
plt.title('Distribución de tipos de cultivos\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de cultivo\n')
plt.ylabel('Cantidad\n')
plt.xticks(rotation=45)
plt.show()

#Conclusión:
#La distribución uniforme de las clases objetivo es un indicador positivo para construir un modelo de predicción sólido, ya que sugiere 
# un conjunto de datos equilibrado y reduce el riesgo de sesgo en las predicciones de nuestro modelo.

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features], orient='h')
plt.title('Diagrama de caja de características numéricas\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Valor de característica\n')
plt.show()

#Conclusión:
#Identificar valores atípicos en las características es crucial y abordarlos se vuelve imperativo si nuestro modelo de predicción 
# encuentra problemas. Los valores atípicos pueden afectar significativamente el rendimiento y la precisión de nuestro modelo, por lo 
# que es necesario considerarlos y manejarlos cuidadosamente para garantizar predicciones sólidas y confiables.

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf")

X = df.drop(columns='Crop')
y= df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#models,predictions = clf.fit(X_train, X_test, y_train, y_test)

#models

#Conclusión
#Con una gran cantidad de clasificadores que ofrecen puntuaciones impresionantes, nos centraremos en los 5 modelos principales para 
# realizar más pruebas y ajustes. Este proceso iterativo nos ayudará a identificar el modelo de predicción más sólido y preciso, 
# garantizando un rendimiento óptimo y el mejor ajuste para nuestro caso de uso específico.

classifiers = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GaussianNB': GaussianNB(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'LGBMClassifier': LGBMClassifier(),
    'BaggingClassifier': BaggingClassifier()
}

param_grid_et = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

results = {}

for name, clf in classifiers.items():
    param_grid = None
    
    if name == 'ExtraTreesClassifier':
        param_grid = param_grid_et
    # Definir param_grid para otros clasificadores de manera similar

    if param_grid is not None:
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)  # Reemplaza X_train, y_train con tus datos
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test)  # Reemplace X_test, y_test con sus datos de prueba
        }


for name, result in results.items():
    print(f"{name}:")
    print(f"  Mejores parámetros: {result['best_params']}")
    print(f"  Mejor puntuación de CV: {result['best_score']:.4f}")
    print(f"  Resultado de la prueba: {result['test_score']:.4f}\n")
    
#Conclusión

cls = ExtraTreesClassifier(max_depth=None, min_samples_split= 2, n_estimators=200)
cls.fit(X_train, y_train)

def evaluate_classifier(cls, X_test, y_test):
    # Hacer predicciones
    y_pred = cls.predict(X_test)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)
    print()
    
    # Precisión de cálculo, recuperación, puntuación F1, soporte
    metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision, recall, f1_score, _ = metrics
    
    print(f"Precisión: {precision:.4f}")
    print(f"Recordar: {recall:.4f}")
    print(f"Puntuación F1: {f1_score:.4f}")
    print()
    
    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud: {accuracy:.4f}")
evaluate_classifier(cls, X_test, y_test)

df['Crop'] = pd.factorize(df['Crop'])[0] + 1
df.head()

#Con nuestros datos meticulosamente limpios, estamos listos para embarcarnos en el viaje revelador del análisis exploratorio de datos 
# (EDA) para descubrir patrones e historias ocultos en su interior.
#Análisis de datos exploratorios
features = list(df.drop(columns='Crop').columns)
data = df.corr()

plt.figure(figsize=(15, 15))
sns.heatmap(data, annot=True, fmt='.2f', cmap='inferno', linewidths=0.5)

plt.title('Mapa de calor de correlación\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Características\n')
plt.ylabel('Características\n')

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

plt.figure(figsize=(15, 15))
sns.countplot(x='Crop', data=df)
plt.title('Distribución de tipos de cultivos\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de cultivo\n')
plt.ylabel('Cantidad\n')
plt.xticks(rotation=45)
plt.show()

#Conclusión:
#La distribución uniforme de las clases objetivo es un indicador positivo para construir un modelo de predicción sólido, ya que sugiere 
# un conjunto de datos equilibrado y reduce el riesgo de sesgo en las predicciones de nuestro modelo.

plt.figure(figsize=(15, 15))
sns.boxplot(data=df[features], orient='h')
plt.title('Diagrama de caja de características numéricas\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Valor de característica\n')
plt.show()

#Conclusión:
#Identificar valores atípicos en las características es crucial y abordarlos se vuelve imperativo si nuestro modelo de predicción 
# encuentra problemas. Los valores atípicos pueden afectar significativamente el rendimiento y la precisión de nuestro modelo, por lo 
# que es necesario considerarlos y manejarlos cuidadosamente para garantizar predicciones sólidas y confiables.

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf")

X = df.drop(columns='Crop')
y= df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#models,predictions = clf.fit(X_train, X_test, y_train, y_test)

#models

#Conclusión
#Con una gran cantidad de clasificadores que ofrecen puntuaciones impresionantes, nos centraremos en los 5 modelos principales para 
# realizar más pruebas y ajustes. Este proceso iterativo nos ayudará a identificar el modelo de predicción más sólido y preciso, 
# garantizando un rendimiento óptimo y el mejor ajuste para nuestro caso de uso específico.

classifiers = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GaussianNB': GaussianNB(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'LGBMClassifier': LGBMClassifier(),
    'BaggingClassifier': BaggingClassifier()
}

param_grid_et = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

results = {}

for name, clf in classifiers.items():
    param_grid = None
    
    if name == 'ExtraTreesClassifier':
        param_grid = param_grid_et
    # Definir param_grid para otros clasificadores de manera similar

    if param_grid is not None:
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)  # Reemplaza X_train, y_train con tus datos
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test)  # Reemplace X_test, y_test con sus datos de prueba
        }


for name, result in results.items():
    print(f"{name}:")
    print(f"  Mejores parámetros: {result['best_params']}")
    print(f"  Mejor puntuación de CV: {result['best_score']:.4f}")
    print(f"  Resultado de la prueba: {result['test_score']:.4f}\n")
    
#Conclusión

cls = ExtraTreesClassifier(max_depth=None, min_samples_split= 2, n_estimators=200)
cls.fit(X_train, y_train)

def evaluate_classifier(cls, X_test, y_test):
    # Hacer predicciones
    y_pred = cls.predict(X_test)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)
    print()
    
    # Precisión de cálculo, recuperación, puntuación F1, soporte
    metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precision, recall, f1_score, _ = metrics
    
    print(f"Precisión: {precision:.4f}")
    print(f"Recordar: {recall:.4f}")
    print(f"Puntuación F1: {f1_score:.4f}")
    print()
    
    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud: {accuracy:.4f}")
evaluate_classifier(cls, X_test, y_test)
