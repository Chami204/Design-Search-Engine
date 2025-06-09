import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

TRAIN_DIR = r"C:\Users\chami.gangoda\OneDrive - Hayleys Group\Desktop\Software creations\CNN model for search engine\model\cnn_shape_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Data pipeline
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/cnn_shape_model.h5")
print("✅ Model trained and saved to model/cnn_shape_model.h5")