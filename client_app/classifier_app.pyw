import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import os
import sys
from torchvision import models
import torch.nn as nn

# Configuration
MODEL_PATH = './classify_best_model3.pth'

# If you had cuda installed use this code instead
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

THRESHOLD = 0.55

# Label mapping
idx2label = {0: 'no_terichomonas gallinaeed', 1: 'terichomonas gallinaeed'}

# Image preprocessing (match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model function
def load_model():

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1))  # <-- fixed here
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

class ClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title('Terichomonas gallinae Image Classifier')

        self.load_btn = tk.Button(master, text='Load Image', command=self.load_image)
        self.load_btn.pack(pady=10)

        self.canvas = tk.Canvas(master, width=224, height=224)
        self.canvas.pack()

        self.classify_btn = tk.Button(master, text='Classify', command=self.classify_image, state=tk.DISABLED)
        self.classify_btn.pack(pady=10)

        self.result_label = tk.Label(master, text='', font=('Arial', 14))
        self.result_label.pack(pady=10)

        self.file_path = None
        self.input_tensor = None

    def load_image(self):
        # Select single image file
        filetypes = [('All Image Files', '*.png *.jpg *.jpeg *.bmp'), ('All Files', '*.*')]
        initialdir = os.path.expanduser(".")  # change to image folder path if needed
        path = filedialog.askopenfilename(title='Select Image', filetypes=filetypes, initialdir=initialdir)


        # Display debug info
        print(f"Selected file: {path}")
        sys.stdout.flush()

        # Load and show image
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            messagebox.showerror('Error', f'Cannot open image: {e}')
            return

        img_resized = img.resize((224, 224))
        self.photo = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Preprocess
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        self.input_tensor = tensor
        self.file_path = path

        self.classify_btn.config(state=tk.NORMAL)
        self.result_label.config(text='')

    def classify_image(self):
        if self.input_tensor is None:
            messagebox.showwarning('Warning', 'Please load an image first')
            return
        with torch.no_grad():
            output = model(self.input_tensor).squeeze(1)
            prob = torch.sigmoid(output).item()
            label = idx2label[int(prob > THRESHOLD)]
            result = f'Result: {label} (p={prob:.2f})'
            print(result)
            sys.stdout.flush()
            self.result_label.config(text=result)

if __name__ == '__main__':
    root = tk.Tk()
    app = ClassifierApp(root)
    root.mainloop()
