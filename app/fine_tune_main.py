import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# === CONFIG ===
# Updated to match your actual file location in the sidebar
working_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(working_dir, "..", "plant_disease_model2_v1.h5")

# Path where your new images are stored
train_data_dir = os.path.join(working_dir, "google_images") 

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10 

# === 1. Load Model with Error Handling ===
try:
    model = load_model(MODEL_PATH)
    print("✅ Original model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # If load_model fails (Python 3.13 bug), you must rebuild architecture as done in main.py
    exit()

# === 2. Freeze Layers (Transfer Learning) ===
# We freeze the early layers to keep the "plant knowledge" 
# and only train the "last layers" on your new images.
for layer in model.layers[:-2]: 
    layer.trainable = False
print("❄️ Base layers frozen.")

# === 3. Prepare Data ===
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

if not os.path.exists(train_data_dir):
    print(f"❌ Folder not found: {train_data_dir}")
    exit()

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Ensure folder count matches model output (38)
)

# === 4. Compile & Train ===
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("model_finetuned.h5", save_best_only=True, monitor="loss")

print("🚀 Starting Fine-Tuning...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("✅ Fine-tuning complete. Saved as model_finetuned.h5")