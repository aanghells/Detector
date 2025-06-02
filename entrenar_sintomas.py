import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import json
import re
import pickle

# --- PREPROCESAMIENTO ---
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# --- CARGAR DATOS DESDE sintomas.json ---
with open("sintomas.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

textos = [limpiar_texto(d["texto"]) for d in datos]
sintomas = [d["sintoma"] for d in datos]

# --- CODIFICAR SÍNTOMAS ---
le = LabelEncoder()
labels = le.fit_transform(sintomas)
num_labels = len(le.classes_)

# --- DIVISIÓN TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(textos, labels, test_size=0.2, random_state=42)

# --- TOKENIZACIÓN ---
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenizar(textos):
    return tokenizer(
        textos,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="tf"
    )

train_encodings = tokenizar(X_train)
test_encodings = tokenizar(X_test)

# Convertir etiquetas a tensores
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# --- MODELO ---
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=num_labels
)

# Compilar modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Entrenar modelo
model.fit(
    x=train_encodings.data,
    y=y_train,
    validation_data=(test_encodings.data, y_test),
    epochs=5,
    batch_size=8
)

# Guardar modelo, tokenizer y label encoder
model.save_pretrained("modelo_sintomas_bert")
tokenizer.save_pretrained("modelo_sintomas_bert")
with open("label_encoder_sintomas.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Modelo BERT entrenado y guardado con éxito.")
print("Síntomas entrenados:", le.classes_)
