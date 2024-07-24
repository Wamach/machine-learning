import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Supongamos que tienes un conjunto de datos X e y
# X = ... (características) train
X = train_data.drop(columns=['id', 'engagement'])  # se elimina la columna por ser la variable objetivo
# y = ... (etiquetas) test
y = train_data['engagement'].astype(int)  # se convierte a entero

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Lista de clasificadores
classifiers = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Evaluar cada clasificador
results = {}
for name, model in classifiers.items():
    # Crear un pipeline para normalización y clasificación
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Validación cruzada con ROC AUC
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    cv_mean_score = cv_scores.mean()

    # Entrenar el modelo en el conjunto de entrenamiento
    pipeline.fit(X_train, y_train)

    # Validar el modelo en el conjunto de validación
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    # Evaluar el modelo en el conjunto de prueba
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Calcular el ROC AUC en el conjunto de prueba
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Almacenar los resultados
    results[name] = {
        "Cross-Validation ROC AUC": cv_mean_score,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "ROC AUC": roc_auc
    }

    # Graficar la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {name}')
    plt.legend(loc='lower right')
    plt.show()

# Mostrar los resultados
results_df = pd.DataFrame(results).T
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.expand_frame_repr', False)  # No truncar la visualización de DataFrames
print(results_df)

# El mejor es:
best_model_name = results_df['Cross-Validation ROC AUC'].idxmax()
best_model = classifiers[best_model_name]
print(f"\nBest Model: {best_model_name}")
