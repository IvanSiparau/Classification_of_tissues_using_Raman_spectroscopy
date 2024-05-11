from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from keras.initializers import glorot_uniform
from keras import regularizers

app = Flask(__name__)

# Загрузка модели PCA
pca_model = 'pca_model.sav'
PCA_model = pickle.load(open(pca_model, 'rb'))

# Загрузка raman_shifts
raman_shifts = np.loadtxt('raman_shifts.txt')

# Загрузка модели нейронной сети
model3_loaded = Sequential()
model3_loaded.add(
    Conv1D(64, 3, activation='relu', input_shape=(32, 1), kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3),
           kernel_initializer=glorot_uniform(seed=42)))
model3_loaded.add(MaxPooling1D(pool_size=2))
model3_loaded.add(BatchNormalization())
model3_loaded.add(Dropout(0.3))

model3_loaded.add(Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3),
                         kernel_initializer=glorot_uniform(seed=42)))
model3_loaded.add(MaxPooling1D(pool_size=2))
model3_loaded.add(BatchNormalization())
model3_loaded.add(Dropout(0.3))

model3_loaded.add(Flatten())

model3_loaded.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_initializer=glorot_uniform(seed=42)))
model3_loaded.add(BatchNormalization())
model3_loaded.add(Dropout(0.5))

model3_loaded.add(Dense(1, activation='sigmoid'))  # Сигмоидальная активационная функция для двух классов

# Загрузка весов
model3_loaded.load_weights('model3_weights.weights.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['file']
    df = pd.read_csv(file, sep="\t", header=None)

    # Извлекаем raman_shift и intensity из DataFrame
    intensity = df[1][1:]
    X = np.array(intensity).reshape(1, -1)
    df = pd.DataFrame(data=X)
    df.columns = raman_shifts
    df = df[raman_shifts[::-1]]

    data = df.copy()

    sklearn_transformed_data = PCA_model.transform(data)
    _PCA = pd.DataFrame(sklearn_transformed_data)

    efs_9 = [2276, 2254, 2282, 2316, 2310, 3273, 2313, 2271, 2322]
    rfi_9 = [2186, 2142, 3537, 2177, 2152, 2149, 3544, 2147, 2185]  # maximum = 3550

    features_efs_rfi = rfi_9 + efs_9

    df_ens = df[features_efs_rfi]

    df = pd.concat([_PCA, df_ens], axis=1, join="inner")

    y_pred = model3_loaded.predict(df)[0]
    print(y_pred)
    color_ = 'red'
    if y_pred[0] < 0.5:
        color_ = '#007bff'
    plt.figure(figsize=(8, 4))
    plt.plot(raman_shifts, intensity, color=color_, linewidth=2)
    plt.xlabel('Raman Shift', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title('Raman Spectrum', fontsize=16)
    plt.grid(True)

    # Сохранение графика в байтовый объект
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Преобразование изображения в формат base64
    graph = base64.b64encode(image_png).decode('utf-8')

    # Результаты классификации
    result = {
        'graph': graph,
        'results': {
            'Раковые клетки': f'{100 * y_pred[0]:.2f}%',
            'Здоровые клетки': f'{100 * (1 - y_pred[0]):.2f}%'
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
