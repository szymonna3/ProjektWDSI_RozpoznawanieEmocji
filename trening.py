import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Sprawdzanie datasetu...")
path = kagglehub.dataset_download("msambare/fer2013")
print("Ścieżka do danych:", path)

train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

if not os.path.exists(train_dir):
    print(f"UWAGA: Nie znaleziono folderu 'train' w {path}")
    print("Zawartość folderu pobranego:", os.listdir(path))
else:
    print(f"Znaleziono dane treningowe w: {train_dir}")

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Wczytywanie obrazów...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nRozpoczynam trening modelu...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64
)

model.save('moj_model_emocji.h5')
print("Model zapisany jako 'moj_model_emocji.h5'!")