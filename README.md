# 🧠 Handwritten Number Recognition

This project implements a CNN-based handwritten digit recognizer trained on the MNIST dataset using **PyTorch** and deployed with a simple **Streamlit** UI. Users can draw a digit (0–9) in the app, and the model will classify it in real-time.

## 📂 Project Structure

├── datasets/  &emsp;&emsp;&ensp;&ensp;&nbsp;   # Contains the MNIST data <br/>
├── models/    &emsp;&emsp;&emsp;&ensp;&nbsp;       # Stores trained model weights <br/>
├── main.ipynb &emsp;&ensp;&nbsp;&ensp;&nbsp;    # Jupyter notebook for training the CNN <br/>
├── inference.py  &emsp;&nbsp;&ensp;&nbsp;   # Streamlit app for digit drawing and recognition <br/>
├── requirements.txt &nbsp; # Required Python dependencies <br/>
├── LICENSE &emsp;&emsp;&emsp;&emsp;# MIT Licence <br/>
└── README.md   &emsp;&emsp;     # You're here <br/>

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- Streamlit

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/suraj-maktala/Handwritten-Number-Recognition.git
cd Handwritten-Number-Recognition
```

### 2. Install dependencies
Install all required packages using pip:
```bash
pip install -r requirements.txt
```

## 💡 How to Use

### Quick Start

To launch the app:
```bash
streamlit run inference.py
```

Several pre-trained models are available in the `models/` directory. By default, the app loads `model_weights_v4.pth`.
To use a different model, simply update the `version` variable in `inference.py` near the bottom of the script
```python
version = "v2"  # Change to v1, v3, v5, etc. as needed
```

### Train the Model Yourself

If you prefer to train your own model:

1. Open the training notebook:
```bash
jupyter notebook main.ipynb
```

2. Train the model from scratch using the MNIST dataset.

3. After training, save the model weights in the `models/` folder with the specified version of choice (e.g., model_weights_v1.pth or model_weights_vx.pth).

4. Then you can run inference using:
```bash
streamlit run inference.py
```

## ✏️ Model Details
- **Architecture**: 2 Conv layers + 2 MaxPool layers + 2 Fully Connected layers
- **Input**: 28×28 grayscale digit
- **Output**: Class probabilities for digits 0–9
- **Performance**: Achieves >97% accuracy on MNIST
