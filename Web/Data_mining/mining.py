import csv
import os
import sys
import pandas as pd
import numpy as np

# Import config
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import config

# Import funciones_auxiliares
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app-web-project'))
sys.path.append(parent_dir)
import funciones_auxiliares

csv.field_size_limit(10**7)

productos = []
nutriscore_numero = []
nutriscore_grado = []
ingredientes_por_producto = []
ingredientes_sin_procesar = []
ingredientes_medio_procesados = []
lista_total_ingredientes_tmp = []
lista_100_max_frec_ingredientes = []
ecoscore_nota = []
ecoscore_grado = []
nova_score = []
foto_producto = []
vectores = []
cnt = 500000
# cnt = 45000
# cnt = 1000000

# Abre el archivo CSV con la codificación UTF-8 y el delimitador de tabulación
with open(config.OPENFOODFACTS_CSV, mode='r', encoding='utf-8') as file:
    lector_csv = csv.reader(file, delimiter='\t')
    
    for linea in lector_csv:
        productos.append(linea[10])
        nutriscore_numero.append(linea[56])
        nutriscore_grado.append(linea[57])
        nova_score.append(linea[58])
        ecoscore_nota.append(linea[68])
        ecoscore_grado.append(linea[69])
        ingredientes_sin_procesar.append(linea[42])
        foto_producto.append(linea[81])

        if cnt < 1:
            break
        cnt -= 1

for e in ingredientes_sin_procesar:
    e = e.split(",")
    ingredientes_medio_procesados.append(e)

def procesar_ingredientes(ingredientes_medio_procesados, ingredientes_por_producto, lista_total_ingredientes):
    for imp in ingredientes_medio_procesados:
        ing_tmp = []
        for ingredientes in imp:
            ingredientes = ingredientes.rsplit(":")
            ingredientes = ingredientes[-1].strip()
            if ingredientes != '':
                lista_total_ingredientes.append(ingredientes)
            ing_tmp.append(ingredientes)
        ingredientes_por_producto.append(ing_tmp)

procesar_ingredientes(ingredientes_medio_procesados, ingredientes_por_producto, lista_total_ingredientes_tmp)

print(len(lista_total_ingredientes_tmp))
print(len(productos))

def sacar_lista_frecuencias(lista_tmp):
    element_count = {}
    top_100_elements = []
    for element in lista_tmp:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    sorted_elements = sorted(element_count.items(), key=lambda x: x[1], reverse=True)
    for element, count in sorted_elements[:100]:
        top_100_elements.append(element)
        print(f"Element: {element}, Frequency: {count}")
    return top_100_elements

lista_100_max_frec_ingredientes = sacar_lista_frecuencias(lista_total_ingredientes_tmp)

funciones_auxiliares.create_vectors_mining_data(lista_100_max_frec_ingredientes, ingredientes_por_producto, vectores)

# Creamos el DataFrame de pandas
df = pd.DataFrame({
    'Productos': productos, 
    'Ingredientes': ingredientes_por_producto, 
    'Nova_Score': nova_score,
    'NutriScore Nota': nutriscore_numero, 
    'Nutriscore Letra': nutriscore_grado,
    'EcoScore Nota': ecoscore_nota, 
    'EcoScore Letra': ecoscore_grado,  
    'Imagenes': foto_producto, 
    'Vectores': vectores
})

# Filter out rows where the value in the "nustriscore" column is "unknown"
df = df[df['Nutriscore Letra'] != 'unknown']
df = df[df['Nutriscore Letra'] != 'not-applicable']
df['Nutriscore Letra'] = df['Nutriscore Letra'].replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(subset=['Nutriscore Letra'])
df = df[df['Nova_Score'] != 'unknown']
df = df[df['Nova_Score'] != 'not-applicable']
df['Nova_Score'] = df['Nova_Score'].replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(subset=['Nova_Score'])


df = df[df['Ingredientes'].apply(len) > 1]

df = df.reset_index(drop=True)
df = df.drop_duplicates(subset=['Productos'], keep=False)

# Count the number of rows with value "b" in the "nutriscore" column
count_a = df[df['Nutriscore Letra'] == 'a'].shape[0]
count_b = df[df['Nutriscore Letra'] == 'b'].shape[0]
count_c = df[df['Nutriscore Letra'] == 'c'].shape[0]
count_d = df[df['Nutriscore Letra'] == 'd'].shape[0]
count_f = df[df['Nutriscore Letra'] == 'e'].shape[0]

# Display the count
print("Number of rows with 'nutriscore' value 'a':", count_a)
print("Number of rows with 'nutriscore' value 'b':", count_b)
print("Number of rows with 'nutriscore' value 'c':", count_c)
print("Number of rows with 'nutriscore' value 'd':", count_d)
print("Number of rows with 'nutriscore' value 'e':", count_f)

test_df = df.tail(50000)
test_df = test_df.reset_index(drop=True)

funciones_auxiliares.exportar_a_csv_data_mining(lista_total_ingredientes_tmp, config.LISTA_TOTAL_INGREDIENTES_CSV)
funciones_auxiliares.exportar_a_csv_data_mining(lista_100_max_frec_ingredientes, config.LISTA_TOP_FREQ_CSV)

# Exportamos el DataFrame a un archivo CSV
test_df.to_csv(config.DATASET_REFERENCIA_PRODUCTOS_CSV, index=False)
df.to_csv(config.DATASET_TESTEO_MODELO_CSV, index=False)
