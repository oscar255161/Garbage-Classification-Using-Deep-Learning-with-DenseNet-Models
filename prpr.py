import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

#saved_state_dict = torch.load('I:\model.pth')
#print(saved_state_dict.keys())


class_names = ['paper', 'glass', 'trash', 'metal', 'plastic', 'cardboard']


num_classes = len(class_names) 


model = models.densenet169(pretrained=False) 
num_ftrs = model.classifier.in_features

model.classifier = nn.Linear(num_ftrs, num_classes)


model.load_state_dict(torch.load('I:\\model.pth'))

model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def classify_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()  


root = tk.Tk()
root.title("PR_Project")
root.geometry("800x800")  

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 16, "bold"), borderwidth='4')


canvas = tk.Canvas(root, width=384, height=512)
canvas.pack(side=tk.TOP, pady=20)


def on_open():
    global image_path, photo_image, canvas
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        image = Image.open(file_path)
        image.thumbnail((384, 512))  
        photo_image = ImageTk.PhotoImage(image)
        canvas.create_image(192, 256, image=photo_image) 

class_index_to_name = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}


def classify_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
        return class_index_to_name[predicted_class_index] 

def on_classify():
    global image_path, label
    if image_path:
        category = classify_image(image_path)
        label.config(text=f"Category: {category}")

def on_clear():
    global image_path, canvas, label
    image_path = None
    canvas.delete("all") 
    label.config(text="Choose Your Image")

button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)


button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=10)

open_button = ttk.Button(button_frame, text="Open", command=on_open, style="TButton")
open_button.pack(side=tk.LEFT, padx=10)

classify_button = ttk.Button(button_frame, text="Start Classification", command=on_classify, style="TButton")
classify_button.pack(side=tk.LEFT, padx=10)

clear_button = ttk.Button(button_frame, text="Clean", command=on_clear, style="TButton")
clear_button.pack(side=tk.LEFT, padx=10)

label = tk.Label(root, text="Choose Your Image", font=("Helvetica", 16), bg="lightblue")
label.pack(side=tk.TOP, fill=tk.X)

root.mainloop()
