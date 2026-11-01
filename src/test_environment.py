import cv2
import torch
import easyocr
import ultralytics
import pandas as pd

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Pandas:", pd.__version__)
print("CUDA available:", torch.cuda.is_available)

print("Environment siap")