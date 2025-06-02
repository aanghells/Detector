import tkinter as tk
from tkinter import ttk
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pyttsx3
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

# Inicializar motor de voz
voz = pyttsx3.init()

# Rutas a modelos y sus codificadores
modelos = {
    "Emociones": {
        "ruta": "modelo_emociones_bert",
        "label_encoder": "label_encoder.pkl"
    },
    "Síntomas": {
        "ruta": "modelo_sintomas_bert",
        "label_encoder": "label_encoder_sintomas.pkl"
    }
}

# Estilos opcionales para emociones
estilos_emociones = {
    "felicidad": {"color": "#FFF9C4", "emoji": "😊"},
    "tristeza": {"color": "#BBDEFB", "emoji": "😢"},
    "amor": {"color": "#F8BBD0", "emoji": "❤️"},
    "ira": {"color": "#FFCDD2", "emoji": "😠"},
    "miedo": {"color": "#D7CCC8", "emoji": "😨"},
    "calma": {"color": "#C8E6C9", "emoji": "😌"},
    "neutro": {"color": "#E0E0E0", "emoji": "😶"},
    "ansiedad": {"color": "#E0AAFF", "emoji": "😰"},
    "nostalgia": {"color": "#8A4932", "emoji": "👵"},
    "motivación": {"color": "#71C55B", "emoji": "😁"},
    "sorpresa": {"color": "#FFA500", "emoji": "🙀"}
}

historial = []
model = None
tokenizer = None
label_encoder = None

def cargar_modelo(nombre_modelo):
    global model, tokenizer, label_encoder
    ruta = modelos[nombre_modelo]["ruta"]
    encoder_file = modelos[nombre_modelo]["label_encoder"]

    model = TFBertForSequenceClassification.from_pretrained(ruta)
    tokenizer = BertTokenizer.from_pretrained(ruta)
    with open(encoder_file, "rb") as f:
        label_encoder = pickle.load(f)

def mostrar_detalle_predicciones(clases, porcentajes):
    ventana_detalle = tk.Toplevel(ventana)
    ventana_detalle.title("📊 Detalles de predicción")
    ventana_detalle.geometry("550x500")
    ventana_detalle.configure(bg="#ffffff")

    tk.Label(ventana_detalle, text="Probabilidades por clase", font=("Helvetica", 12, "bold"), bg="#ffffff").pack(pady=10)

    for clase, porc in zip(clases, porcentajes):
        texto = f"{clase.capitalize()}: {round(porc * 100, 2)}%"
        tk.Label(ventana_detalle, text=texto, font=("Helvetica", 11), bg="#ffffff").pack(anchor="w", padx=20)

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    colores = [estilos_emociones.get(c, estilos_emociones["neutro"])["color"] for c in clases]
    ax.bar(clases, porcentajes, color=colores)
    ax.set_ylabel('Probabilidad')
    ax.set_title('Distribución')
    ax.set_ylim(0, 1)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=ventana_detalle)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

def predecir():
    texto = entrada.get()
    if not texto.strip():
        resultado.config(text="⚠️ Escribe algo para analizar")
        return

    inputs = tokenizer(texto, return_tensors="tf", padding=True, truncation=True, max_length=50)
    outputs = model(inputs)[0]
    probs = tf.nn.softmax(outputs, axis=1).numpy()[0]

    idx = np.argmax(probs)
    pred_clase = label_encoder.inverse_transform([idx])[0]
    porcentaje = round(probs[idx] * 100, 2)

    estilo = estilos_emociones.get(pred_clase, estilos_emociones["neutro"])
    emoji = estilo["emoji"]
    color = estilo["color"]

    ventana.configure(bg=color)
    fondo_frame.configure(bg=color)
    titulo.configure(bg=color)
    resultado.configure(
        bg=color,
        text=f"{emoji} Detectado: {pred_clase.capitalize()} ({porcentaje}%)"
    )

    voz.say(f"Se detectó: {pred_clase}, con {porcentaje} por ciento de certeza")
    voz.runAndWait()

    registro = f"[{datetime.now().strftime('%H:%M:%S')}] '{texto}' → {pred_clase.capitalize()} ({porcentaje}%)"
    historial.append(registro)
    lista_historial.insert(tk.END, registro)

    clases = label_encoder.classes_
    mostrar_detalle_predicciones(clases, probs)

def guardar_historial():
    if historial:
        with open("historial.txt", "w", encoding="utf-8") as f:
            for linea in historial:
                f.write(linea + "\n")
    ventana.destroy()

# --- Crear ventana principal ---
ventana = tk.Tk()
ventana.title("🧠 Detector Inteligente")
ventana.geometry("520x520")
ventana.configure(bg="#f2f2f2")

fuente_titulo = ("Helvetica", 16, "bold")
fuente_normal = ("Helvetica", 12)

fondo_frame = tk.Frame(ventana, bg="#f2f2f2")
fondo_frame.pack(expand=True, fill="both", padx=10, pady=10)

titulo = tk.Label(fondo_frame, text="🧠 Detector de Texto", font=fuente_titulo, bg="#f2f2f2", fg="#333")
titulo.pack(pady=10)

# Menú de selección de modelo
tk.Label(fondo_frame, text="Selecciona modelo:", font=fuente_normal, bg="#f2f2f2").pack()
opcion_modelo = tk.StringVar(value="Emociones")
selector = ttk.Combobox(fondo_frame, textvariable=opcion_modelo, values=list(modelos.keys()), state="readonly")
selector.pack(pady=5)

def cargar_seleccion():
    cargar_modelo(opcion_modelo.get())
    resultado.config(text="✅ Modelo cargado correctamente")

btn_cargar = tk.Button(fondo_frame, text="Cargar modelo", command=cargar_seleccion, bg="#2196F3", fg="white", font=fuente_normal)
btn_cargar.pack(pady=10)

tk.Label(fondo_frame, text="Escribe una frase:", font=fuente_normal, bg="#f2f2f2").pack()
entrada = tk.Entry(fondo_frame, font=fuente_normal, width=40, justify="center")
entrada.pack(pady=8)

btn_detectar = tk.Button(fondo_frame, text="Detectar", font=fuente_normal, bg="#4CAF50", fg="white", command=predecir)
btn_detectar.pack(pady=10)

resultado = tk.Label(fondo_frame, text="😶 Esperando entrada...", font=("Helvetica", 14), bg="#f2f2f2", fg="#0066cc")
resultado.pack(pady=15)

tk.Label(fondo_frame, text="📋 Historial:", font=("Helvetica", 11), bg="#f2f2f2").pack()
lista_historial = tk.Listbox(fondo_frame, width=60, height=7, font=("Helvetica", 10))
lista_historial.pack(pady=10)

ventana.protocol("WM_DELETE_WINDOW", guardar_historial)
ventana.mainloop()
