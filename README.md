# 🧠 Handwritten Number Recognition

This project implements a CNN-based handwritten digit recognizer trained on the MNIST dataset using **PyTorch** and deployed with a simple **Streamlit** UI. Users can draw a digit (0–9) in the app, and the model will classify it in real-time.

## 📂 Project Structure

├── datasets/  &emsp;&emsp;&ensp;&ensp;&nbsp;   # Contains the MNIST data (optional if using torchvision) <br/>
├── models/    &emsp;&emsp;&emsp;&ensp;&nbsp;       # Stores trained model weights <br/>
├── main.ipynb &emsp;&ensp;&nbsp;&ensp;&nbsp;    # Jupyter notebook for training the CNN <br/>
├── inference.py  &emsp;&nbsp;&ensp;&nbsp;   # Streamlit app for digit drawing and recognition <br/>
├── requirements.txt &nbsp; # Required Python dependencies <br/>
└── README.md   &emsp;&emsp;     # You're here <br/>

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- Streamlit
