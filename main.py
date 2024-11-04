import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split = 0.2
)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\pythonProject\pollution-sea-dataset\pollution-sea-dataset\datasetGarbage/train',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
    validation_split=0.2,  # 20% para validação e 80% treinamento
    subset="training",      # Para o conjunto de treinamento
    seed=123               # Para reprodutibilidade
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\pythonProject\pollution-sea-dataset\pollution-sea-dataset\datasetGarbage/validation',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',

)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\pythonProject\pollution-sea-dataset\pollution-sea-dataset\datasetGarbage/test',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
)

# Definir um modelo básico (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  #  uma camada de Dropout
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes / garbage e no garbage
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Treinar o modelo usando o conjunto de treinamento e validação
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

model.save('main.h5')
