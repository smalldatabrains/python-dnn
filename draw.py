import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw an MNIST Digit")
        
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")

    def predict_digit(self):
        # Resize image to 28x28 pixels
        image = self.image.resize((28, 28))
        # Invert image (white background, black digit)
        image = ImageOps.invert(image)
        # Convert to numpy array
        image = np.array(image)
        # Normalize the image
        image = image / 255.0
        # Reshape to fit model input
        image = image.reshape(1, 28, 28, 1)

        # Load your pre-trained model
        model = None

        # Make a prediction
        prediction = model.predict(image)
        digit = np.argmax(prediction)

        # Display the prediction
        print(f"Predicted Digit: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
