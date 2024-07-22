# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset en un DataFrame de Pandas
df = pd.read_excel('ENB2012_DATA.xlsx')

# Considerar la columna Y2 como la variable objetivo
y = df['Y2']

# En este caso Y1 y Y2, son las variables determinantes, así que se eliminan.
x = df.drop(columns=['Y1', 'Y2', 'X8'])  # Se elimina la columna X8 por ser categorica,
# Y1 y Y2 por ser las variables objetivo

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Se generan los métodos de regresión a utilizarse (Lineal, Polinomial, knn y árbol de decisión)
modelos = {
    'Regresión Lineal': Pipeline([
        ('estandarizador', StandardScaler()), # Se estandarizan los datos
        ('regresor', LinearRegression()) # Se aplica la regresión lineal
    ]),
    'Regresión Polinomial': Pipeline([
        ('estandarizador', StandardScaler()),
        ('polinomio', PolynomialFeatures(degree=2)),
        ('regresor', LinearRegression())
    ]),
    'KNN': Pipeline([
        ('estandarizador', StandardScaler()),
        ('regresor', KNeighborsRegressor())
    ]),
    'Árbol de Decisión': Pipeline([
        ('estandarizador', StandardScaler()),
        ('regresor', DecisionTreeRegressor())
    ])
}

# Entrenar y evaluar los modelos
resultados = {}
metricas = {}
for nombre, modelo in modelos.items():
    modelo.fit(x_train, y_train)  # Se entrena modelo
    y_pred = modelo.predict(x_test)  # Se evalua el modelo
    r2 = r2_score(y_test, y_pred)  # Se calcula el R2
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Se calcula el RMSE
    resultados[nombre] = {'R2': r2, 'RMSE': rmse}
    metricas[nombre] = cross_val_score(modelo, x, y, cv=10, scoring='r2')
    # Se aplica cross validation con un criterio de r2

# Regularizacion o Normalizacion usando Ridge o Lasso
# from sklearn.linear_model import Ridge, Lasso


# Imprimir resultados
for nombre, scores in resultados.items():
    print(f'{nombre} - R2: {scores["R2"]:.4f}, RMSE: {scores["RMSE"]:.4f}')
    print(f'Cross Validation R2: {metricas[nombre].mean():.4f} (+/- {metricas[nombre].std() * 2:.4f})')

# Seleccionar el mejor modelo basado en R2
mejor_modelo = max(resultados, key=lambda k: resultados[k]['R2'])
print(f'\nMejor modelo basado en R2: {mejor_modelo}')



# Grafica de los valores reales vs los valores predichos
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=3)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.title('Valores Reales vs Valores Predichos')
plt.show()

# Realizar predicciones con el mejor modelo -- Complemento para visualizarlo mejor
y_pred = modelos[mejor_modelo].predict(x_test)
print('Valor real\tValor predicho')
for real, pred in zip(y_test, y_pred):
    print(f'{real:.2f}\t{pred:.2f}')
