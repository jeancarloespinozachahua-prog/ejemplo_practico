import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image, ImageOps, ImageDraw

# Cargar el modelo guardado y el escalador
try:
    modelo = tf.keras.models.load_model("modelo_digitos.keras")
    escalador = joblib.load("escalador_digitos.pkl")
except Exception as e:
    messagebox.showerror("Error", "Primero debes ejecutar 'entrenar_modelo.py' para generar los archivos del modelo.")
    exit()

class AppDibujoDigitos:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Interactivo de Reconocimiento de Dígitos - SENATI")
        self.root.geometry("680x520")
        self.root.config(bg="#f0f2f5")

        # Título
        titulo = tk.Label(root, text="Reconocimiento de Dígitos a Mano Alzada", font=("Arial", 15, "bold"), bg="#f0f2f5", fg="#1a73e8")
        titulo.pack(pady=10)

        # Marco principal
        main_frame = tk.Frame(root, bg="#f0f2f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Panel izquierdo: Lienzo para dibujar
        left_frame = tk.LabelFrame(main_frame, text=" Panel de Dibujo ", font=("Arial", 10, "bold"), bg="#ffffff", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="black", cursor="cross")
        self.canvas.pack(pady=5)
        self.canvas.bind("<B1-Motion>", self.dibujar)

        # Imagen PIL de respaldo para trazos en memoria
        self.image_pil = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_pil = ImageDraw.Draw(self.image_pil)

        # Botones
        btn_frame = tk.Frame(left_frame, bg="#ffffff")
        btn_frame.pack(fill=tk.X, pady=5)

        btn_limpiar = tk.Button(btn_frame, text="Limpiar", bg="#ea4335", fg="white", font=("Arial", 9, "bold"), command=self.limpiar_canvas)
        btn_limpiar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_predecir = tk.Button(btn_frame, text="Predecir", bg="#34a853", fg="white", font=("Arial", 9, "bold"), command=self.predecir_dibujo)
        btn_predecir.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        # Panel derecho: Resultados y Probabilidades
        right_frame = tk.LabelFrame(main_frame, text=" Resultados y Probabilidades ", font=("Arial", 10, "bold"), bg="#ffffff", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_resultado = tk.Label(right_frame, text="Predicción: -", font=("Arial", 13, "bold"), bg="#ffffff", fg="#ea4335")
        self.lbl_resultado.pack(anchor="w", pady=5)

        tk.Label(right_frame, text="Probabilidades por clase:", bg="#ffffff", font=("Arial", 9, "bold")).pack(anchor="w", pady=2)
        
        self.txt_probabilidades = tk.Text(right_frame, height=9, width=30, font=("Consolas", 8), bg="#f8f9fa")
        self.txt_probabilidades.pack(fill=tk.BOTH, expand=True, pady=5)

    def dibujar(self, event):
        r = 12
        x, y = event.x, event.y
        # Dibujar en el Canvas de Tkinter
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        # Dibujar en la imagen de PIL
        self.draw_pil.ellipse([x - r, y - r, x + r, y + r], fill="white", outline="white")

    def limpiar_canvas(self):
        self.canvas.delete("all")
        self.image_pil = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw_pil = ImageDraw.Draw(self.image_pil)
        self.lbl_resultado.config(text="Predicción: -")
        self.txt_probabilidades.delete("1.0", tk.END)

    def predecir_dibujo(self):
        try:
            # Redimensionar la imagen a 8x8 píxeles exactos (igual que load_digits)
            img_resized = self.image_pil.resize((8, 8), Image.Resampling.LANCZOS)
            
            # Convertir a arreglo numpy y reescalar al rango 0-16
            arr = np.array(img_resized, dtype=float)
            arr = arr / 255.0 * 16.0
            arr = arr.flatten().reshape(1, -1)

            # Escalar con el StandardScaler
            arr_escalado = escalador.transform(arr)

            # Predecir con el modelo Keras
            probabilidades = modelo.predict(arr_escalado, verbose=0)[0]
            clase_predicha = np.argmax(probabilidades)

            # Mostrar resultados
            self.lbl_resultado.config(text=f"Predicción: Dígito {clase_predicha}")

            self.txt_probabilidades.delete("1.0", tk.END)
            for i, prob in enumerate(probabilidades):
                barra = "█" * int(prob * 12)
                self.txt_probabilidades.insert(tk.END, f"Dígito {i}: {prob*100:5.1f}% {barra}\n")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al procesar el dibujo:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppDibujoDigitos(root)
    root.mainloop()