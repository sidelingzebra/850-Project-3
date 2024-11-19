import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os
from ultralytics import YOLO


model = YOLO("best.pt")
results1 = model.predict("ardmega.jpg", save=True, imgsz=928)
results2 = model.predict("arduno.jpg", save=True, imgsz=928)
results3 = model.predict("rasppi.jpg", save=True, imgsz=928)
