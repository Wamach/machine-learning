import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# Cargar el archivo CSV
file_path = "typed_uanl.csv"
df = pd.read_csv(file_path)

# Entidades del conjunto de datos.
"""
Entidades en el conjunto de datos:
Nombre
Sueldo Neto
dependencia
Fecha
Tipo
"""

print("\nEntidades en el conjunto de datos:")
print(df.columns)

# separar en entidades para obtenr la estadistica descriptiva de cada entidad
# Nombre
print("\nEstadistica descriptiva de la entidad Nombre:")
print(df["Nombre"].describe())

# Sueldo Neto
print("\nEstadistica descriptiva de la entidad Sueldo Neto:")
print(df["Sueldo Neto"].describe())

# dependencia
print("\nEstadistica descriptiva de la entidad dependencia:")
print(df["dependencia"].describe())

# Fecha
print("\nEstadistica descriptiva de la entidad Fecha:")
print(df["Fecha"].describe())

# Tipo
print("\nEstadistica descriptiva de la entidad Tipo:")
print(df["Tipo"].describe())

# Se hace agrupacion por ano 2019, 2020, 2021, 2022, 2023
# 2019
df_2019 = df[df["Fecha"].str.contains("2019")]
df_2020 = df[df["Fecha"].str.contains("2020")]
df_2021 = df[df["Fecha"].str.contains("2021")]
df_2022 = df[df["Fecha"].str.contains("2022")]
df_2023 = df[df["Fecha"].str.contains("2023")]

# Crear intervalos de sueldos
bins = list(range(0, 150000, 10000))  # Intervalos de 10000 pesos
df['Sueldo Neto Rango'] = pd.cut(df['Sueldo Neto'], bins=bins)

# Contar la frecuencia de sueldos en cada rango
sueldo_rango_counts = df['Sueldo Neto Rango'].value_counts().sort_index()

# Gráfico de barras para la frecuencia de sueldos en rangos específicos
plt.figure(figsize=(12, 8))
ax = sueldo_rango_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frecuencia de Sueldos Netos por Rango de 10000 Pesos')
plt.xlabel('Rango de Sueldo Neto (Pesos)')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir el número total de empleados por encima de cada barra
for i in ax.containers:
    ax.bar_label(i, label_type='edge')

# Guardar y mostrar el gráfico
plt.savefig("frecuencia_sueldos_netos_rango.png", bbox_inches='tight')
plt.show()


# En una grafica de pastel, proporcion de Tipo de empleados
df["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados")
plt.savefig("proporcion_tipo_empleados.png")
plt.show()


# grafica de pastel, para cada año, la proporcion de tipo de empleados
df_2019["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados en 2019")
plt.savefig("proporcion_tipo_empleados_2019.png")
plt.show()

df_2020["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados en 2020")
plt.savefig("proporcion_tipo_empleados_2020.png")
plt.show()

df_2021["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados en 2021")
plt.savefig("proporcion_tipo_empleados_2021.png")
plt.show()

df_2022["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados en 2022")
plt.savefig("proporcion_tipo_empleados_2022.png")
plt.show()

df_2023["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de empleados en 2023")
plt.savefig("proporcion_tipo_empleados_2023.png")
plt.show()

# Suma de los sueldos netos por cada tipo en cada año
# Una vez que se tiene la suma, se hacee un histogram para cada año
# y despues un grafico que muestre la evolucion de todos los años para ver la comparativa

# 2019
df_2019_sum = df_2019.groupby("Tipo")["Sueldo Neto"].sum()
df_2019_sum.plot.bar()
plt.title("Suma de sueldos netos por tipo de empleados en 2019")
plt.savefig("suma_sueldos_netos_2019.png")
plt.show()

# 2020
df_2020_sum = df_2020.groupby("Tipo")["Sueldo Neto"].sum()
df_2020_sum.plot.bar()
plt.title("Suma de sueldos netos por tipo de empleados en 2020")
plt.savefig("suma_sueldos_netos_2020.png")
plt.show()


# 2021
df_2021_sum = df_2021.groupby("Tipo")["Sueldo Neto"].sum()
df_2021_sum.plot.bar()
plt.title("Suma de sueldos netos por tipo de empleados en 2021")
plt.savefig("suma_sueldos_netos_2021.png")
plt.show()

# 2022
df_2022_sum = df_2022.groupby("Tipo")["Sueldo Neto"].sum()
df_2022_sum.plot.bar()
plt.title("Suma de sueldos netos por tipo de empleados en 2022")
plt.savefig("suma_sueldos_netos_2022.png")
plt.show()

# 2023
df_2023_sum = df_2023.groupby("Tipo")["Sueldo Neto"].sum()
df_2023_sum.plot.bar()
plt.title("Suma de sueldos netos por tipo de empleados en 2023")
plt.savefig("suma_sueldos_netos_2023.png")
plt.show()

# Combinar los resultados en un DataFrame
sueldo_tipo_año = pd.DataFrame({
    '2019': df_2019_sum,
    '2020': df_2020_sum,
    '2021': df_2021_sum,
    '2022': df_2022_sum,
    '2023': df_2023_sum
}).T

# Crear el gráfico de barras apiladas
sueldo_tipo_año.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')

# Mejorar el aspecto del gráfico
plt.title('Evolución de Sueldos Netos por Tipo de Empleados')
plt.xlabel('Año')
plt.ylabel('Sueldo Neto (Pesos)')
plt.xticks(rotation=0)
plt.legend(title='Tipo de Empleado', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.savefig("evolucion_sueldos_netos_tipo_empleados.png")
plt.show()

# Histograma de los sueldos netos de los empleados de la UANL por dependencia
df["dependencia"].value_counts().plot.bar()
plt.title("Sueldos netos por dependencia")
plt.savefig("sueldos_netos_dependencia.png")
plt.show()

# Se genera otra agrupacion de: sueldos netos por dependencia y año
# 2019
df_2019_dep = df_2019.groupby("dependencia")["Sueldo Neto"].sum()
df_2019_dep.plot.bar()
plt.title("Suma de sueldos netos por dependencia en 2019")
plt.savefig("suma_sueldos_netos_dependencia_2019.png")
plt.show()

# 2020
df_2020_dep = df_2020.groupby("dependencia")["Sueldo Neto"].sum()
df_2020_dep.plot.bar()
plt.title("Suma de sueldos netos por dependencia en 2020")
plt.savefig("suma_sueldos_netos_dependencia_2020.png")
plt.show()

# 2021
df_2021_dep = df_2021.groupby("dependencia")["Sueldo Neto"].sum()
df_2021_dep.plot.bar()
plt.title("Suma de sueldos netos por dependencia en 2021")
plt.savefig("suma_sueldos_netos_dependencia_2021.png")
plt.show()

# 2022
df_2022_dep = df_2022.groupby("dependencia")["Sueldo Neto"].sum()
df_2022_dep.plot.bar()
plt.title("Suma de sueldos netos por dependencia en 2022")
plt.savefig("suma_sueldos_netos_dependencia_2022.png")
plt.show()

# 2023
df_2023_dep = df_2023.groupby("dependencia")["Sueldo Neto"].sum()
df_2023_dep.plot.bar()
plt.title("Suma de sueldos netos por dependencia en 2023")
plt.savefig("suma_sueldos_netos_dependencia_2023.png")
plt.show()

# Agrupacion de sueldo neto, dependencia y tipo
# 5 empleados con mayor sueldo neto
# Porcentaje de dependencia y tipo que tiene esta mayor proporcion.
df_dep_tipo = df.groupby(["dependencia", "Tipo"])["Sueldo Neto"].sum()
df_dep_tipo = df_dep_tipo.reset_index()
df_dep_tipo = df_dep_tipo.sort_values(by="Sueldo Neto", ascending=False)
top_5 = df_dep_tipo.head(5)

# Grafica de pastel por dependencia y tipo de los 5 empleados con mayor sueldo neto
top_5["dependencia"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de dependencia de los 5 empleados con mayor sueldo neto")
plt.savefig("proporcion_dependencia_top_5.png")
plt.show()

top_5["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Proporcion de tipo de los 5 empleados con mayor sueldo neto")
plt.savefig("proporcion_tipo_top_5.png")
plt.show()

# El salario mas alto de cada Tipo
df_tipo_max = df.groupby("Tipo")["Sueldo Neto"].max()
df_tipo_max.plot.bar()
plt.title("Salario mas alto por tipo de empleado")
plt.savefig("salario_mas_alto_tipo_empleado.png")
plt.show()

# El salario mas bajo de cada Tipo
df_tipo_min = df.groupby("Tipo")["Sueldo Neto"].min()
df_tipo_min.plot.bar()
plt.title("Salario mas bajo por tipo de empleado")
plt.savefig("salario_mas_bajo_tipo_empleado.png")
plt.show()

# Muestrame si el nombre de una persona en el mismo año se repite y cuantes veces por medio de una grafica
# Muestrame en una grafica de pastel el porcentajes de veces que se repite mas 2 o mas veces el nombre de una persona
# diferenciandolo por tipo
df_nombre = df.groupby(["Nombre", "Fecha", "Tipo"])["Sueldo Neto"].count()
df_nombre = df_nombre.reset_index()
df_nombre = df_nombre.sort_values(by="Sueldo Neto", ascending=False)
df_nombre = df_nombre[df_nombre["Sueldo Neto"] > 1]

df_nombre["Tipo"].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Porcentaje de veces que se repite un nombre mas de 1 vez")
plt.savefig("porcentaje_nombre_repetido.png")
plt.show()

# En una grafica de bigotes muestrame la relacion entre el sueldo neto y el tipo de empleado
df.boxplot(column="Sueldo Neto", by="Tipo")
plt.title("Relacion entre sueldo neto y tipo de empleado")
plt.savefig("relacion_sueldo_tipo_empleado.png")
plt.show()

# Agrupaciones que se realizaron:
# sueldo_tipo_año
# df_2019_dep, df_2020_dep, df_2021_dep, df_2022_dep, df_2023_dep
# df_dep_tipo
# top_5
# df_tipo_max
# df_tipo_min
# df_nombre

# Obtener estadisticas de cada agrupacion:
print("\nEstadistica descriptiva de la agrupacion sueldo_tipo_año:")
print(sueldo_tipo_año.describe())

print("\nEstadistica descriptiva de la agrupacion df_2019_dep:")
print(df_2019_dep.describe())

print("\nEstadistica descriptiva de la agrupacion df_2020_dep:")
print(df_2020_dep.describe())

print("\nEstadistica descriptiva de la agrupacion df_2021_dep:")
print(df_2021_dep.describe())

print("\nEstadistica descriptiva de la agrupacion df_2022_dep:")
print(df_2022_dep.describe())

print("\nEstadistica descriptiva de la agrupacion df_2023_dep:")
print(df_2023_dep.describe())

print("\nEstadistica descriptiva de la agrupacion df_dep_tipo:")
print(df_dep_tipo.describe())

print("\nEstadistica descriptiva de la agrupacion top_5:")
print(top_5.describe())

print("\nEstadistica descriptiva de la agrupacion df_tipo_max:")
print(df_tipo_max.describe())

print("\nEstadistica descriptiva de la agrupacion df_tipo_min:")
print(df_tipo_min.describe())

print("\nEstadistica descriptiva de la agrupacion df_nombre:")
print(df_nombre.describe())

# Seleccionar una agrupacion para realizar prueba ANOVA, e identificar si hay diferencia entre los elementos
# Verificar  si las muestras son o no normales
# Si son normales: Prueba de ANOVA
# Si no son normales: Prueba de Kruskal-Wallis y Tukey para saber quien es el diferente

# Pruebas de normalidad para la agrupacion df_2020_sum
print("\nPruebas de normalidad para la agrupacion df_2020_sum:")
shapiro_test_df_2020_sum = shapiro(df_2020_sum.values.flatten())
print("Shapiro-Wilk test p-value:", shapiro_test_df_2020_sum.pvalue)

if shapiro_test_df_2020_sum.pvalue > 0.05:
    print("\nPrueba ANOVA para la agrupacion df_2020_sum:")
    # Para ANOVA, necesitamos datos en formato largo, no en sumas, así que usamos df_2020
    model_df_2020_sum = ols('Q("Sueldo Neto") ~ C(Tipo)', data=df_2020).fit()
    anova_table_df_2020_sum = sm.stats.anova_lm(model_df_2020_sum, typ=2)
    print(anova_table_df_2020_sum)

    if anova_table_df_2020_sum["PR(>F)"].iloc[0] < 0.05:
        print("\nPrueba de comparaciones múltiples (t-student) en la agrupacion df_2020_sum:")
        tipos = df_2020["Tipo"].unique()
        for tipo1, tipo2 in combinations(tipos, 2):
            grupo1 = df_2020[df_2020["Tipo"] == tipo1]["Sueldo Neto"]
            grupo2 = df_2020[df_2020["Tipo"] == tipo2]["Sueldo Neto"]
            ttest_result = ttest_ind(grupo1, grupo2)
            print(f"Comparación {tipo1} vs {tipo2}: p-value = {ttest_result.pvalue}")
else:
    print("\nPrueba de Kruskal-Wallis para la agrupacion df_2020_sum:")
    kruskal_test_df_2020_sum = kruskal(*[group["Sueldo Neto"].values for name, group in df_2020.groupby("Tipo")])
    print("Kruskal-Wallis test p-value:", kruskal_test_df_2020_sum.pvalue)

    if kruskal_test_df_2020_sum.pvalue < 0.05:
        print("\nPrueba de Tukey para comparaciones múltiples en la agrupacion df_2020_sum:")
        tukey_result_df_2020_sum = pairwise_tukeyhsd(df_2020["Sueldo Neto"], df_2020["Tipo"])
        print(tukey_result_df_2020_sum)

# Pruebas de normalidad para la agrupacion sueldo_tipo_año
print("\nPruebas de normalidad para la agrupacion sueldo_tipo_año:")
shapiro_test_sueldo_tipo_año = shapiro(sueldo_tipo_año.values.flatten())
print("Shapiro-Wilk test p-value:", shapiro_test_sueldo_tipo_año.pvalue)

if shapiro_test_sueldo_tipo_año.pvalue > 0.05:
    print("\nPrueba ANOVA para la agrupacion sueldo_tipo_año:")
    # Para ANOVA, necesitamos datos en formato largo, no en sumas, así que usamos el DataFrame original y filtramos
    df_sueldo_tipo_año_long = df[df["Fecha"].str.contains("2019|2020|2021|2022|2023")]
    model_sueldo_tipo_año = ols('Q("Sueldo Neto") ~ C(Tipo) * C(Año)', data=df_sueldo_tipo_año_long).fit()
    anova_table_sueldo_tipo_año = sm.stats.anova_lm(model_sueldo_tipo_año, typ=2)
    print(anova_table_sueldo_tipo_año)

    if anova_table_sueldo_tipo_año["PR(>F)"].iloc[0] < 0.05:
        print("\nPrueba de comparaciones múltiples (t-student) en la agrupacion sueldo_tipo_año:")
        tipos = df_sueldo_tipo_año_long["Tipo"].unique()
        for tipo1, tipo2 in combinations(tipos, 2):
            grupo1 = df_sueldo_tipo_año_long[df_sueldo_tipo_año_long["Tipo"] == tipo1]["Sueldo Neto"]
            grupo2 = df_sueldo_tipo_año_long[df_sueldo_tipo_año_long["Tipo"] == tipo2]["Sueldo Neto"]
            ttest_result = ttest_ind(grupo1, grupo2)
            print(f"Comparación {tipo1} vs {tipo2}: p-value = {ttest_result.pvalue}")
else:
    print("\nPrueba de Kruskal-Wallis para la agrupacion sueldo_tipo_año:")
    kruskal_test_sueldo_tipo_año = kruskal(*[group["Sueldo Neto"].values for name, group in df.groupby("Tipo")])
    print("Kruskal-Wallis test p-value:", kruskal_test_sueldo_tipo_año.pvalue)

    if kruskal_test_sueldo_tipo_año.pvalue < 0.05:
        print("\nPrueba de Tukey para comparaciones múltiples en la agrupacion sueldo_tipo_año:")
        tukey_result_sueldo_tipo_año = pairwise_tukeyhsd(df["Sueldo Neto"], df["Tipo"])
        print(tukey_result_sueldo_tipo_año)

# En total se generaron 25 graficas
# De los cuales 10 son de pastel, 5 de barras, 5 de bigotes, 5 de histogramas
