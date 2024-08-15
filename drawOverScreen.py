import tkinter as tk
import pyautogui

class AreaSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)  # Pantalla completa
        self.root.attributes('-alpha', 0.5)  # Ventana semi-transparente con opacidad del 50%
        self.root.configure(bg='black')
        self.root.attributes('-topmost', True)  # Mantener al frente
        # self.root.wm_attributes("-transparentcolor", "black")
        
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Añadir texto en el canvas, centrado con margen superior
        self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, 
            self.root.winfo_screenheight() // 2 - 8,
            anchor="n", 
            text="Seleccione un área de interés", 
            font=("Helvetica", 16), 
            fill="white"
        )

        self.start_x, self.start_y = None, None
        self.rect_id = None

        # Enlazar eventos
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Variable para almacenar las coordenadas
        self.coordinates = {}

    def on_mouse_down(self, event):
        # Captura las coordenadas absolutas del mouse
        self.start_x, self.start_y = pyautogui.position()
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=3)

    def on_mouse_drag(self, event):
        if self.rect_id:
            # Captura las coordenadas actuales del mouse
            current_x, current_y = pyautogui.position()
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, current_x, current_y)

    def on_mouse_up(self, event):
        if self.rect_id:
            # Captura las coordenadas finales del mouse
            end_x, end_y = pyautogui.position()
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, end_x, end_y)
            # Guardar las coordenadas en un diccionario
            self.coordinates = {
                "x1": self.start_x,
                "y1": self.start_y,
                "x2": end_x,
                "y2": end_y
            }
            self.root.quit()  # Cierra la ventana

    def get_coordinates(self):
        self.root.mainloop()  # Mostrar la ventana y esperar la selección
        self.root.destroy()
        return self.coordinates  # Retornar las coordenadas seleccionadas
    
