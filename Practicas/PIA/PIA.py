import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tabulate import tabulate

# Datos a utilizar
datos_entrenamiento = pd.read_csv('train.csv')

# Preparar características y etiquetas
X = datos_entrenamiento.drop(columns=['id', 'engagement'])
y = datos_entrenamiento['engagement'].astype(int)  # Se castea a entero

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_temp, y_entrenamiento, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Lista de clasificadores - Modelos
clasificadores = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', random_state=42),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001,
                                    max_iter=2000, random_state=42),
    "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, learning_rate=1.0, algorithm="SAMME", random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
}

# Evaluar cada modelo / clasificador
resultados = {}
for nombre, modelo in clasificadores.items():
    # Se crea objeto Pipeline para escalar los datos y utilizar los clasificadores (modelos)
    pipeline = Pipeline([
        ('escalador', StandardScaler()),
        ('clasificador', modelo)
    ])

    try:
        # Validación cruzada con ROC AUC y con cv=10
        puntuaciones_cv = cross_val_score(pipeline, X_entrenamiento, y_entrenamiento, cv=10, scoring='roc_auc')
        puntuacion_media_cv = puntuaciones_cv.mean()

        # Almacenar los resultados en un diccionario para cada modelo
        resultados[nombre] = {
            "ROC AUC Validación Cruzada": puntuacion_media_cv
        }

        # Evaluar el modelo en el conjunto de prueba para obtener la curva ROC basada en validación cruzada
        pipeline.fit(X_entrenamiento, y_entrenamiento)
        y_probabilidad_predicha_cv = cross_val_score(pipeline, X_prueba, y_prueba, cv=10, scoring='roc_auc').mean()
        roc_auc_cv = y_probabilidad_predicha_cv

        # Graficar la curva ROC si el ROC AUC de validación cruzada es mayor a 0.75
        if puntuacion_media_cv > 0.75:
            fpr, tpr, umbrales = roc_curve(y_prueba, pipeline.predict_proba(X_prueba)[:, 1])
            plt.figure()
            plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc_cv:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title(f'Curva Característica (ROC) - {nombre}')
            plt.legend(loc='lower right')
            plt.show()

    except AttributeError as e:
        print(f"Ocurrió un error al intentar graficar la curva ROC para el modelo {nombre}: {e}")
        continue

# Mostrar los resultados en una tabla
resultados_df = pd.DataFrame(resultados).T

# Usar tabulate para imprimir la tabla de manera más legible
print(tabulate(resultados_df, headers='keys', tablefmt='psql'))

# El mejor modelo
nombre_mejor_modelo = resultados_df['ROC AUC Validación Cruzada'].idxmax()
mejor_modelo = clasificadores[nombre_mejor_modelo]
print(
    f"\nMejor Modelo: {nombre_mejor_modelo} con ROC AUC Validación Cruzada: {resultados_df.loc[nombre_mejor_modelo, 'ROC AUC Validación Cruzada']}")

# Gráfica para comparar los modelos de forma visual
colores = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink', 'lightyellow', 'lightblue',
           'lightgray', 'lightgoldenrodyellow', 'lightsteelblue', 'lightcyan', 'lightseagreen', 'lightcoral',
           'lightpink']

plt.figure(figsize=(12, 8))

# Graficar las barras con etiquetas y colores personalizados
barras = plt.bar(resultados_df.index, resultados_df['ROC AUC Validación Cruzada'],
                 yerr=resultados_df['ROC AUC Validación Cruzada'].std(),
                 capsize=5, color=colores[:len(resultados_df)], edgecolor='grey')

# Añadir la línea de referencia
plt.axhline(y=0.75, color='r', linestyle='--', linewidth=2, label='Umbral 0.75')

# Añadir las etiquetas en las barras
for barra in barras:
    yval = barra.get_height()
    plt.text(barra.get_x() + barra.get_width() / 2.0, yval + 0.01, round(yval, 2),
             ha='center', va='bottom', fontsize=12)

# Añadir las etiquetas y el título
plt.ylabel('ROC AUC', fontsize=14)
plt.xlabel('Modelos de Clasificación', fontsize=14)
plt.title('ROC AUC Validación Cruzada de Modelos de Clasificación', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Añadir la leyenda
plt.legend(loc='lower right', fontsize=12)

# Añadir un grid para facilitar la lectura
plt.grid(axis='y', linestyle='--', linewidth=0.7)

# Ajustar los márgenes
plt.tight_layout()

# Mostrar la gráfica
plt.show()


