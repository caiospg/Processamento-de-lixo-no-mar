import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = tf.keras.models.load_model('main.h5')

# Dicionário para mapear classes a labels
class_labels = {0: 'Resíduo Sólido', 1: 'Rio/MAR Limpo'}

def classificar_imagem(img_path):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma nova dimensão para o batch

    # Normalizar a imagem
    img_array /= 255.0

    # Fazer a predição
    predicao = model.predict(img_array)
    classe_predita = np.argmax(predicao, axis=1)[0]  # Pega o índice da classe com maior probabilidade

    predicao = model.predict(img_array)
    print(f'Probabilidades: {predicao}')  # Isso mostrará as probabilidades para cada classe

    # Resultado
    return class_labels[classe_predita]

# Usar a função
caminho_imagem = r'C:\Users\Dudu\Desktop\Reconhecimento\Processamento-de-lixo-no-mar\pollution-sea-dataset\pollution-sea-dataset\datasetGarbage\train\Garbage/image_125.jpg'

resultado = classificar_imagem(caminho_imagem)
print(f'A imagem classificada é: {resultado}')

# (Opcional) Mostrar a imagem
img = image.load_img(caminho_imagem)
plt.imshow(img)
plt.axis('off')  # Não mostrar eixos
plt.show()