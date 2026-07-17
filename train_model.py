import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Importing libraries...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pickle

print("Checking dataset folder...")

BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_PATH):
    print("❌ ERROR: 'dataset' folder not found!")
    print("👉 Run: python prepare_dataset.py")
    exit()

print(f"✅ Dataset found at: {DATASET_PATH}")

print("Preparing data...")

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Building model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 🔥 PARTIAL UNFREEZE (VERY IMPORTANT)
for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

# 🔥 Callbacks (VERY IMPORTANT)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 🔥 Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

print("Evaluating model...")
model.evaluate(val_data)

# Save final model
model.save("cataract_model.h5")

# Save class mapping
pickle.dump(train_data.class_indices, open("class_indices.pkl", "wb"))

print("✅ Training complete!")