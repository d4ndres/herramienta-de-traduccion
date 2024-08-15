import tkinter as tk

class TextDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)  # Pantalla completa
        self.root.attributes('-topmost', True)  # Mantener al frente
        self.root.wm_attributes("-transparentcolor", self.root['bg'])  # Transparente

        # Configurar el canvas para ocupar todo el espacio
        self.canvas = tk.Canvas(self.root, bg=self.root['bg'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Crear un Label para el texto con fondo negro y texto blanco
        self.label = tk.Label(
            self.canvas, 
            text="Texto inicial", 
            font=("Helvetica", 16), 
            fg="white", 
            bg="black"
        )
        self.label.place(relx=0.5, rely=0.2, anchor="center")  # Ubicarlo en el centro de la pantalla

    def update_text(self, new_text):
        # Actualiza el texto del Label
        self.label.config(text=new_text)
        self.root.update()

