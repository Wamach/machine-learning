# Librerías
import os

# Establece el número de threads a 3
os.environ['OMP_NUM_THREADS'] = '3'
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Cargar y preparar el conjunto de datos
df = pd.read_excel('ENB2012_data.xlsx')
y = df['X6']
X = df.drop(columns=['X6', 'Y1', 'Y2'])

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
    'Logistic Regression': LogisticRegression(max_iter=500, solver='liblinear')
    # Se usa el solver liblinear para evitar warnings
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

# Calcular métricas de evaluación con zero_division ajustado
print(f"\nMejor Modelo: {best_model_name}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ---------------- Tarea 4 ----------------
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Suma de las Distancias Cuadradas')
plt.title('Elbow Method - Metodo del Codo')
plt.show()

# Determinar el número óptimo de clusters basado en el método del codo
optimal_k = 2  # Valor optimo para la grafica de codo

# Reajustar el modelo K-means con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X_train)
y_pred_kmeans = kmeans.predict(X_test)

# Mostrar graficamente los resultados de Kmeans utilizando t-SNE para reducir la dimensionalidad
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
X_test_tsne = tsne.fit_transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_tsne[y_pred_kmeans == 0, 0], X_test_tsne[y_pred_kmeans == 0, 1], color='red', label='Cluster 1')
plt.scatter(X_test_tsne[y_pred_kmeans == 1, 0], X_test_tsne[y_pred_kmeans == 1, 1], color='blue', label='Cluster 2')
plt.title('K-means Clustering con t-SNE')
plt.legend()
plt.show()

silhouette_avg = silhouette_score(X_test, y_pred_kmeans) # Se utiliza la silueta para evaluar el agrupamiento
print("Silhouette coefficient:", silhouette_avg)

if accuracy_score(y_test, y_pred) > 0.8:
    print("El modelo de clasificacion sobrepasa el umbral")
elif silhouette_avg > 0.4:
    print("El modelo de agrupamiento sobrepasa el umbral")

# Definir umbrales
accuracy_threshold = 0.8
silhouette_threshold = 0.4
# Evaluar el modelo de agrupamiento
if silhouette_avg > silhouette_threshold:
    print("El modelo de agrupamiento es mejor")
else:
    print("El modelo de clasificacion es mejor")