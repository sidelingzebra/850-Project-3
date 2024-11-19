import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")



results = model.train(data='datasets/data.yaml', epochs=1,imgsz=928,batch=16,
                      plots=True)
model.export()




