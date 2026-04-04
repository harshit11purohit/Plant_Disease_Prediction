import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# === CONFIG ===
MODEL_PATH = r"C:\Users\Jagdish singh\OneDrive\Desktop\project\app\trained_model\plant_disease_prediction_model.h5"


  # Path to your trained model
FINE_TUNE_DIR = 'app/google_images'  # Folder where your Google images are stored
IMG_SIZE = (224, 224)  # Update if your model uses different size
BATCH_SIZE = 4
EPOCHS = 5  # Few epochs since small dataset

# === Load your existing model ===
model = load_model(MODEL_PATH)
print("Loaded existing model.")

# === Freeze base layers (optional) ===
# You can skip this if you want to train all layers
for layer in model.layers[:-1]:  # Freeze all except last
    layer.trainable = False
print("Base layers frozen.")

# === Prepare image data ===
datagen = ImageDataGenerator(rescale=1./255)

train_data_dir = os.path.abspath("google_images")
train_generator = datagen.flow_from_directory(train_data_dir,     target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# === Optional: Check number of classes ===
print("Classes detected:", train_generator.class_indices)

# === Recompile the model ===
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Save the best fine-tuned model ===
checkpoint = ModelCheckpoint("model_finetuned.h5", save_best_only=True, monitor="loss")

# === Fine-tune ===
model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("Fine-tuning complete. Saved as model_finetuned.h5")
