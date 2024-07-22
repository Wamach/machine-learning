# Librerias
import os
os.environ['OMP_NUM_THREADS'] = '3'  # Establece el número de threads a 3

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Cargar y preparar el conjunto de datos
df = pd.read_excel('ENB2012_data.xlsx')
y = df['X6']
X = df.drop(columns=['X6','Y1','Y2'])

# Escalar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear varios modelos de clasificación
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=500, solver='liblinear') # Se usa el solver liblinear para evitar warnings
}

# Aplicar Cross Validation y evaluar los modelos
results = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    results[model_name] = cv_scores
    print(f"{model_name} Cross Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Seleccionar el mejor modelo basado en la media de las puntuaciones de validación cruzada
best_model_name = max(results, key=lambda k: results[k].mean())
best_model = models[best_model_name]

# Entrenar el mejor modelo en el conjunto de entrenamiento completo
best_model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calcular métricas de evaluación
print(f"\nMejor Modelo: {best_model_name}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Mostrar graficamente
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [result.mean() for result in results.values()], yerr=[result.std() for result in results.values()], capsize=5)
plt.ylabel('Accuracy')
plt.title('Grafico de Barras. Cross Validation Accuracy de los Modelos de Clasificación')
plt.show()