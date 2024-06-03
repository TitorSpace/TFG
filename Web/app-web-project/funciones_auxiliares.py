import csv
import numpy as np
import pandas as pd
import ast
from Levenshtein import ratio


def exportar_a_csv(lista_strings, nombre_archivo):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for string in lista_strings:
            escritor_csv.writerow([string])

def importar_csv(nombre_archivo):
    lista_strings = []
    with open(nombre_archivo, mode='r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        for fila in lector_csv:
            lista_strings.extend(fila)
    return lista_strings


def buscar_productos(nombre_producto, archivo_csv):
    # Leer el archivo CSV
    df = pd.read_csv(archivo_csv)
    
    # Buscar el producto en la columna "Productos"
    producto = df[df['Productos'] == nombre_producto]
    
    # Si el producto se encuentra, devolver la lista de ingredientes
    if not producto.empty:
        ingredientes_str = producto['Ingredientes'].values[0]
        # Usar ast.literal_eval para convertir la cadena a una lista
        ingredientes = ast.literal_eval(ingredientes_str)
        return ingredientes
    else:
        return None



def buscar_ingredientes_similares(query, lista_ingredientes):
    similares = []
    query = query.lower()
    for ingrediente in lista_ingredientes:
        # Compara la similitud y considera como coincidencia si el ratio es mayor a 0.5
        if ratio(query, ingrediente.lower()) > 0.5:
            similares.append(ingrediente)
    return similares

def buscar_productos_similares(query, lista_productos):
    similares = []
    query = query.lower()
    for ingrediente in lista_productos:
        # Compara la similitud y considera como coincidencia si el ratio es mayor a 0.5
        if ratio(query, ingrediente.lower()) > 0.5:
            similares.append(ingrediente)
    return similares


import csv

def leer_csv_y_obtener_productos(ruta_csv):
    try:
        productos = []
        with open(ruta_csv, newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv)
            # Iteramos sobre cada fila del archivo CSV
            for fila in lector_csv:
                # Agregamos el producto de la columna alista
                productos.append(fila[0])
        return productos
    except FileNotFoundError:
        print("El archivo CSV no fue encontrado.")
        return []
    except IndexError:
        print("El formato del archivo CSV no es el esperado.")
        return []

# Ejemplo de uso:
# productos = leer_csv_y_obtener_productos('ruta/archivo.csv')
# print(productos)


def prepararDatosNutriScore(output):
    output.columns = ['E', 'D', 'C', 'B', 'A']
    return output.idxmax(axis=1)[0]
    


def prepararDatosNovaScore(output):
    output.columns = ['4', '3', '2', '1']
    return output.idxmax(axis=1)[0]
    
    
def procesarPrediccion(outputData):
    if outputData.shape[1]==5:
        return prepararDatosNutriScore(outputData)
    else:
        return prepararDatosNovaScore(outputData)


def predecir_resultados(producto, modelo):
    # Realiza cualquier preprocesamiento necesario en el producto
    # Por ejemplo, puedes convertir el producto en un vector o realizar otras transformaciones
    
    # Aquí, supondremos que el producto es una lista de características (features)
    x = producto.reshape(1,-1)
    # Realiza la predicción utilizando el modelo cargado
    resultado_prediccion = modelo.predict(x)
    
    return resultado_prediccion


def create_vectors_from_input(lista_total_ingredientes, ing_por_producto):
    vector = np.zeros(len(lista_total_ingredientes), dtype=int)
    for i, ingrediente in enumerate(lista_total_ingredientes):
        if ingrediente in ing_por_producto:
            vector[i] = 1
    return vector


def create_vectors_mining_data(lista_total_ingredientes, ingredientes_por_producto, vectores):
    for i in range(len(ingredientes_por_producto)):
        tmp_vector=[]
        for j in range(len(lista_total_ingredientes)):
            if lista_total_ingredientes[j] in ingredientes_por_producto[i]:
                tmp_vector.append(1)
            else:
                tmp_vector.append(0)
        vectores.append(tmp_vector)

def exportar_a_csv_data_mining(lista_strings, nombre_archivo):
    with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for string in lista_strings:
            escritor_csv.writerow([string])

def importar_csv_data_mining(nombre_archivo):
    lista_strings = []
    with open(nombre_archivo, mode='r', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        for fila in lector_csv:
            lista_strings.extend(fila)
    return lista_strings