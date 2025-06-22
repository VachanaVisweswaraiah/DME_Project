import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam

# Load labels
df = pd.read_csv("labels.csv")

# Train/Val split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Image generators
img_size = 150
batch_size = 32
seed = 42

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_dataframe(
    train_df,
    directory="train",
    x_col="filename",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    seed=seed
)
val_generator = val_gen.flow_from_dataframe(
    val_df,
    directory="train",
    x_col="filename",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    seed=seed
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
label_to_index = train_generator.class_indices
class_weight_dict = {
    label_to_index[label]: weight
    for label, weight in zip(np.unique(df['label']), class_weights)
}
num_classes = len(label_to_index)

# Load MobileNetV2
base_model = MobileNetV2(input_shape=(img_size, img_size, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = True  # fine-tune entire model

# Build classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=12, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint("models/dme_model_mobilenet.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
]

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Accuracy Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("reports/accuracy_plot_mobilenet.png")
plt.show()

# Classification Report
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - MobileNetV2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("reports/confusion_matrix_mobilenet.png")
plt.show()
