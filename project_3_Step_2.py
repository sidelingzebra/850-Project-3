import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


 
results = model.train(data='datasets/data.yaml', epochs=180,imgsz=928,batch=8,
                      plots=True,save_period=10,name="150")
model.export()




results1 = model.predict("ardmega.jpg", save=True, imgsz=928)
results2 = model.predict("arduno.jpg", save=True, imgsz=928)
results3 = model.predict("rasppi.jpg", save=True, imgsz=928)
