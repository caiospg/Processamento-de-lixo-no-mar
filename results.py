import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = tf.keras.models.load_model('main.h5')

# Dicionário para mapear classes a labels
class_labels = {0: 'Resíduo Sólido', 1: 'Rio/MAR Limpo', 2: 'Manchas de oleo'}

def classificar_imagem(img_path):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma nova dimensão para o batch

    img_array /= 255.0

    predicao = model.predict(img_array)

    print(f'Probabilidades: {predicao}')

    classe_predita = np.argmax(predicao, axis=1)[0]  # Pega o índice da classe com maior probabilidade

    return class_labels[classe_predita]

# Usar a função
caminho_imagem = 'C:\caio\Processamento-de-lixo-no-mar\pollution-sea-dataset\datasetGarbage/test/Non Garbage/No_Oil_Spill00996.jpg'

resultado = classificar_imagem(caminho_imagem)
print(f'A imagem classificada é: {resultado}')

# (Opcional) Mostrar a imagem
img = image.load_img(caminho_imagem)
plt.imshow(img)
plt.axis('off')  # Não mostrar eixos
plt.show()