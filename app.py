import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

class_names = ['1 BD', '10 BD', '100fils', '20 BD', '25fils', '5 BD', '500 fils', '50fils']

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model_small.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img):
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    i = prob.argmax().item()
    return class_names[i], prob[i].item()

st.title("Bahraini Coins Classifier")
option = st.sidebar.radio("chose way", ["upload photo", "camera"])

if option == "upload photo":
    file = st.file_uploader("choese photo", type=["jpg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=300)
        name, conf = predict(img)
        st.success(f"{name} ({conf*100:.1f}%)")

else:
    run = st.checkbox("on cam")
    screen = st.empty()
    cap = cv2.VideoCapture(0)

    while run:
        ok, frame = cap.read()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        name, conf = predict(Image.fromarray(rgb))
        result.success(f"{name} ({conf*100}%)")
        screen.image(rgb)
    cap.release()
