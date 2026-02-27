from ultralytics import YOLO
import torch
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    device=0
)

model.save("best.pt")
