import os
import sys
from flask import Flask, jsonify, redirect, render_template, request, url_for 
import pandas as pd
import tensorflow as tf
import funciones_auxiliares
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config

app = Flask(__name__)

model_nutri = tf.keras.models.load_model(config.MODELO_GRANDE_NUTRISCORE_H5)
model_nova = tf.keras.models.load_model(config.MODELO_GRANDE_NOVASCORE_H5)
lista_max_freq = funciones_auxiliares.importar_csv(config.LISTA_TOP_FREQ_CSV)
lista_productos = funciones_auxiliares.leer_csv_y_obtener_productos(config.LISTA_PRODUCTOS_CSV)

ingredientes_prod = []
current_vector = []
current_nova_score = ""
current_nutri_score = ""

@app.route('/index')
def home():
    return render_template('index.html', ingredientes_prod=ingredientes_prod)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/add', methods=['POST'])
def add_value():
    value = request.form['valueInput']
    ingredientes_prod.append(value)
    return redirect(url_for('home'))

@app.route('/delete/<int:index>', methods=['POST'])
def delete_value(index):
    if index < len(ingredientes_prod):
        ingredientes_prod.pop(index)
    return redirect(url_for('home'))

@app.route('/predecir', methods=['POST'])
def predecir():
    if not ingredientes_prod:
        error_message = "You must add at least one ingredient"
        return render_template('index.html', ingredientes_prod=ingredientes_prod, error=error_message)
    global current_vector, current_nova_score, current_nutri_score
    vector = funciones_auxiliares.create_vectors_from_input(lista_max_freq, ingredientes_prod)
    current_vector = vector
    predic_nova = funciones_auxiliares.predecir_resultados(vector, model_nova)
    predict_nutri = funciones_auxiliares.predecir_resultados(vector, model_nutri)
    output_nova = funciones_auxiliares.procesarPrediccion(pd.DataFrame(predic_nova))
    output_nutri = funciones_auxiliares.procesarPrediccion(pd.DataFrame(predict_nutri))
    current_nova_score = output_nova
    current_nutri_score = output_nutri
    return render_template('index.html', resultado_nova=f'Nova Score value: {output_nova}', 
                           resultados_nutri=f'NutriScore value: {output_nutri}')

@app.route('/suggest_ingredient', methods=['GET'])
def suggest_ingredient():
    text = request.args.get('term', '')
    suggestions = funciones_auxiliares.buscar_ingredientes_similares(text, lista_max_freq)
    return jsonify(suggestions)

@app.route('/reset', methods=['POST'])
def reset():
    del ingredientes_prod[:]
    return redirect(url_for('home'))

@app.route('/buscar_producto', methods=['POST'])
def buscar_producto():
    producto = request.form['productoInput']
    ingredientes = funciones_auxiliares.buscar_productos(producto, config.DATASET_REFERENCIA_PRODUCTOS_CSV)
    if ingredientes:
        ingredientes_prod.extend(ingredientes)
    else:
        error_message = f"Product '{producto}' not found"
        return render_template('index.html', ingredientes_prod=ingredientes_prod, error=error_message)
    return redirect(url_for('home'))

@app.route('/suggest_product', methods=['GET'])
def suggest_product():
    text = request.args.get('term', '')
    suggestions = funciones_auxiliares.buscar_productos_similares(text, lista_productos)
    return jsonify(suggestions)

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form['feedback']
    nova_score_correction = request.form.get('novaScoreCorrection', "")
    nutri_score_correction = request.form.get('nutriScoreCorrection', "")

    # Leer el archivo CSV si existe
    if os.path.exists(config.FEEDBACK_CSV_PATH):
        df = pd.read_csv(config.FEEDBACK_CSV_PATH)
    else:
        df = pd.DataFrame(columns=['vector', 'nova_scores', 'nutri_scores'])

    vector_str = str(current_vector)
    if vector_str in df['vector'].values:
        # Actualizar las listas de valores si el vector ya existe
        idx = df.index[df['vector'] == vector_str].tolist()[0]
        nova_scores = eval(df.at[idx, 'nova_scores'])
        nutri_scores = eval(df.at[idx, 'nutri_scores'])

        if feedback == 'no':
            if nova_score_correction:
                nova_scores.append(nova_score_correction)
                df.at[idx, 'nova_scores'] = str(nova_scores)
            if nutri_score_correction:
                nutri_scores.append(nutri_score_correction)
                df.at[idx, 'nutri_scores'] = str(nutri_scores)
            message = "Thank you for your correction!"
        else:
            message = "Thank you for your feedback!"
    else:
        if feedback == 'no':
            nova_scores = [nova_score_correction] if nova_score_correction else []
            nutri_scores = [nutri_score_correction] if nutri_score_correction else []
            new_row = pd.DataFrame({
                'vector': [vector_str],
                'nova_scores': [str(nova_scores)],
                'nutri_scores': [str(nutri_scores)]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            message = "Thank you for your correction!"
        else:
            message = "Thank you for your feedback!"

    # Guardar el DataFrame actualizado en el archivo CSV
    df.to_csv(config.FEEDBACK_CSV_PATH, index=False)

    return render_template('index.html', ingredientes_prod=ingredientes_prod, resultado_nova=f'Nova Score value: {current_nova_score}', 
                           resultados_nutri=f'NutriScore value: {current_nutri_score}', feedback_message=message)

if __name__ == "__main__":
    app.run(debug=True)
