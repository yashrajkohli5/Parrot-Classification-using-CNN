import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("parrot.h5")

class_names = ['amazon green parrot', 'gray parrot', 'macaw', 'white parrot']
# class_labels = {0 :'amazon green parrot', 1 :'gray parrot', 2 :'macaw', 3 :'white parrot'}

img_size = 224
def predict_image(img_path):
  img = image.load_img(img_path, target_size = (img_size, img_size))
  img_arr = image.img_to_array(img)/255.0
  img_arr = np.expand_dims(img_arr, axis = 0)

  pred = model.predict(img_arr)
  class_index = np.argmax(pred)
  confidence = np.max(pred)

  return class_names[class_index], confidence

def open_image():
  global panel

  file_path = filedialog.askopenfilename(
      filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
  )

  if not file_path:
    return

  # Load and display image
  img = Image.open(file_path)
  img = img.resize((250,250))
  img_tk = ImageTk.PhotoImage(img)


  panel.configure(image = img_tk)
  panel.image = img_tk

  # Predict
  label, conf = predict_image(file_path)
  result_label.config(
      text = f"Prediction: {label}\nConfidence: {conf:.2f}"
  )
  
  
root = tk.Tk()
root.title("Parrot Species Classifier (CNN Model)")
root.geometry("400x450")
root.configure(bg = "#E8F1F5")

title_label = tk.Label(root, text = "Parrot Classification", font = ("Arial", 18, "bold"), bg = "#E8F1F5")
title_label.pack(pady = 10)

panel = tk.Label(root)
panel.pack(pady = 10)

btn = tk.Button(root, text = "Upload Parrot Image", command = open_image, font = ("Arial", 14), bg = "#5DADE2", fg = "white")
btn.pack(pady = 10)

result_label = tk.Label(root, text = "", font = ("Arial", 14), bg = "#E8F1F5")
result_label.pack(pady = 20)

root.mainloop()