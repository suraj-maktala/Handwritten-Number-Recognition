from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import center_of_mass
import torch
import torch.nn as nn


### Device Definition ###
device = "cuda" if torch.cuda.is_available() else "cpu"


### Model Definition ###
OUTPUT_CHANNELS = 4
KERNEL_SIZE = (3,3)
HIDDEN_UNITS_1 = 512
HIDDEN_UNITS_2 = 64
class Number_Recognition_Model(nn.Module):
    def __init__(self, input_features, output_labels):
        super().__init__()
        
        C, H, W = input_features
        self.conv_layer_1 = nn.Conv2d(in_channels=C, out_channels=OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE) 
        self.relu_layer = nn.ReLU()
        self.maxpool_layer_1 = nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride = (1,1))
        self.conv_layer_2 = nn.Conv2d(in_channels=OUTPUT_CHANNELS, out_channels=2*OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE)
        self.maxpool_layer_2 = nn.MaxPool2d(kernel_size=KERNEL_SIZE)
        self.flatten_layer = nn.Flatten()
        self.linear_layer_1 = nn.Linear(in_features=392, out_features=HIDDEN_UNITS_2)
        self.activation_layer = nn.LeakyReLU()
        # self.linear_layer_2 = nn.Linear(in_features=HIDDEN_UNITS_1, out_features=HIDDEN_UNITS_2)
        self.linear_layer_final = nn.Linear(in_features=HIDDEN_UNITS_2, out_features=output_labels) 
    
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.relu_layer(x)
        x = self.maxpool_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.relu_layer(x)
        x = self.maxpool_layer_2(x)
        x = self.flatten_layer(x)
        x = self.linear_layer_1(x)
        x = self.activation_layer(x)
        # x = self.linear_layer_2(x)
        # x = self.activation_layer(x)
        logits = self.linear_layer_final(x)

        return logits

### Model Instantiation ###
def load_model():
    input_features = (1,28,28)
    output_labels = 10
    model = Number_Recognition_Model(input_features, output_labels)
    model = model.to(device)

    return model

### Region of Interest ###
def get_roi_bounds(img):
    # Find all non-zero pixel indices
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)

    if not rows.any() or not cols.any():
        return None  # No foreground found

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return y_min, y_max, x_min, x_max

version = "v4"
model = load_model()
model.load_state_dict(torch.load(f"models/model_weights_{version}.pth", weights_only=True))


### StreamLit Code ###
st.title("Digit Recognizer 🎨🧠")
st.write("Draw a digit (0-9) below:")

# --- Drawing canvas --- #
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,               # Increased thickness
    stroke_color="white",          # White digit
    background_color="black",      # Black background
    height=240, width=240,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Prediction logic --- #
if canvas_result.image_data is not None:
    
    img = canvas_result.image_data
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # --- Calculate Center of mass and center adjust the image --- #
    cy, cx = center_of_mass(img)
    actual = (img.shape[1]//2,img.shape[0]//2)
    diffx = actual[0] - cx
    diffy = actual[1] - cy

    if not np.isnan(diffx) and diffx >= 1:
        bound = int(diffx)
        img = img[:,:-bound]
        img = np.pad(img,pad_width=((0,0),(bound,0)), mode="constant", constant_values=0)

    elif not np.isnan(diffx):
        bound = int(diffx)
        img = img[:, -bound:]
        img = np.pad(img,pad_width=((0,0),(0,-bound)), mode="constant", constant_values=0)
    
    if not np.isnan(diffy) and diffy >= 1:
        bound = int(diffy)
        img = img[:-bound, :]
        img = np.pad(img,pad_width=((bound,0),(0,0)), mode="constant", constant_values=0)
    
    elif not np.isnan(diffy):
        bound = int(diffy)
        img = img[-bound:, :]
        img = np.pad(img,pad_width=((0,-bound),(0,0)), mode="constant", constant_values=0)

    # --- Calculate Region of Interest and adjust image accordingly --- #
    if img is not None:
        val = get_roi_bounds(img)

        if val is not None:
            y1, y2, x1, x2 = get_roi_bounds(img)
            thresh = min(
                abs(y2-img.shape[0]),
                abs(y1),
                abs(x2-img.shape[1]),
                abs(x1)
            )

            img = img[thresh:-thresh, thresh:-thresh]
            img = np.pad(img, pad_width=20, mode="constant",constant_values=0)
        
        # --- Resize the Image --- #
        img = img / 255.0
        
        img = cv2.resize(img, (28,28))
        _, img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)
    
        st.image(img, caption="Processed 28x28 Image",width=150)

        img = img.reshape(1,1,28,28)
        img = img.astype(np.float32)

        # --- Prediction --- #
        predict = st.button("Predict")
        if predict:
            with torch.inference_mode():
                img = torch.from_numpy(img).to(device=device)
                model.eval()
                output = torch.nn.functional.softmax(model(img.to(device)), dim=1)
                pred = torch.argmax(output).item()
                st.success(f"🧠 Predicted Digit: {pred}")