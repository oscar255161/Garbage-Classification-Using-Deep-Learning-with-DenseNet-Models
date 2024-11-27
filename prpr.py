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

# 定义类别数量
num_classes = len(class_names)  # 确保这里是您的类别数量

# 创建 DenseNet169 模型实例
model = models.densenet169(pretrained=False)  # 使用 pretrained=False，因为我们将加载自己的权重
num_ftrs = model.classifier.in_features

# 修改分类器以匹配您的类别数量
model.classifier = nn.Linear(num_ftrs, num_classes)

# 加载保存的权重
model.load_state_dict(torch.load('I:\\model.pth'))



# 将模型设置为评估模式
model.eval()







# 预处理函数
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

# 预测函数
def classify_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()  # 需要将索引映射到类别名称

# 创建 GUI
root = tk.Tk()
root.title("PR_Project")
root.geometry("800x800")  # 设置窗口大小

# 创建 Style 对象
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 16, "bold"), borderwidth='4')

# 创建 Canvas 组件来显示图片
canvas = tk.Canvas(root, width=384, height=512)
canvas.pack(side=tk.TOP, pady=20)


def on_open():
    global image_path, photo_image, canvas
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        image = Image.open(file_path)
        image.thumbnail((384, 512))  # 调整图像大小
        photo_image = ImageTk.PhotoImage(image)
        canvas.create_image(192, 256, image=photo_image)  # 将图像放置在 Canvas 的中心
# 类别索引到类别名称的映射
class_index_to_name = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# 预测函数
def classify_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class_index = predicted.item()
        return class_index_to_name[predicted_class_index]  # 将索引映射到类别名称

def on_classify():
    global image_path, label
    if image_path:
        category = classify_image(image_path)
        label.config(text=f"Category: {category}")

def on_clear():
    global image_path, canvas, label
    image_path = None
    canvas.delete("all")  # 清除 Canvas
    label.config(text="Choose Your Image")
# 创建一个 frame 用于放置按钮
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

# UI 组件
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