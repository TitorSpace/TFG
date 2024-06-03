import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
# Ensure the parent directory is in the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config

def load_and_preprocess_data(filepath,score):
    """Load and preprocess data."""
    data = pd.read_csv(filepath)
    # Convert string representations of vectors to numerical arrays
    data['Vectores'] = data['Vectores'].apply(lambda x: np.array(eval(x)))
    # Handle missing values
    data.dropna(inplace=True)
    # Split dataset into features and target
    X = np.stack(data['Vectores'].values)
    y = pd.get_dummies(data[score]).values
    # Normalize input data
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y

def build_model(input_dim, output_dim):
    """Build and compile the model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def cross_validate_model(X, y, k_folds=5, epochs=50, batch_size=128):
    """Perform cross-validation and return accuracy scores."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracy_scores = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
        print(f'Fold {fold+1}/{k_folds}')
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model = build_model(X.shape[1], y.shape[1])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        accuracy_scores.append(val_acc)
        print(f'Validation Accuracy: {val_acc}')
    
    return accuracy_scores, model

def main():
    novascore="Nova_Score"
    nutriscore="Nutriscore Letra"
    X_nova, y_nova = load_and_preprocess_data(config.DATASET_TESTEO_MODELO_CSV,novascore)
    X_nutri, y_nutri = load_and_preprocess_data(config.DATASET_TESTEO_MODELO_CSV,nutriscore)

    accuracy_scores_nova, model_nova = cross_validate_model(X_nova, y_nova)
    accuracy_scores_nutri, model_nutri = cross_validate_model(X_nutri, y_nutri)


    # Calculate and print average validation accuracy
    average_accuracy_nova = np.mean(accuracy_scores_nova)
    average_accuracy_nutri = np.mean(accuracy_scores_nutri)
    print(f'Average Validation NovaScore Accuracy: {average_accuracy_nova}')
    print(f'Average Validation NutriScore Accuracy: {average_accuracy_nutri}')

    # Save the model
    model_nutri.save(config.MODELO_GRANDE_NUTRISCORE_H5)
    model_nova.save(config.MODELO_GRANDE_NOVASCORE_H5)
    print(f'Model NutriScore saved to {config.MODELO_GRANDE_NUTRISCORE_H5}')
    print(f'Model NovaScore saved to {config.MODELO_GRANDE_NOVASCORE_H5}')

if __name__ == "__main__":
    main()