import tkinter as tk
from PIL import ImageGrab, Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_cnn.h5")

# Initialize Tkinter
root = tk.Tk()
root.title("Handwritten Digit Recognition")
canvas_width = 200
canvas_height = 200
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

# Variables to store mouse position
last_x, last_y = None, None

# Draw function
def draw(event):
    global last_x, last_y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, event.x, event.y,
                           width=12, fill='black', capstyle=tk.ROUND, smooth=True)
    last_x, last_y = event.x, event.y

def reset_position(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_position)

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="")

# Predict function
def predict_digit():
    # Capture canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1))

    # Preprocess image
    img = img.convert('L')                # grayscale
    img = ImageOps.invert(img)            # invert colors (MNIST style)
    img = img.resize((28,28))             # resize
    img_array = np.array(img)/255.0       # normalize
    img_array = img_array.reshape(1,28,28)

    # Predict digit
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    result_label.config(text=f"Predicted Digit: {digit}")

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Predict", command=predict_digit).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side=tk.LEFT, padx=5)

root.mainloop()
