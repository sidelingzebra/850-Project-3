import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pathlib
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


 
results = model.train(data='datasets/data.yaml', epochs=150,imgsz=928,batch=16,
                      plots=True,save_period=10,name="150")
model.export()




results1 = model.predict("ardmega.jpg", save=True, imgsz=928)
results2 = model.predict("arduno.jpg", save=True, imgsz=928)
results3 = model.predict("rasppi.jpg", save=True, imgsz=928)



# metrics = model.val()

# model("ardmega.jpg")
# model("arduno.jpg")
# model("rasppi.jpg")

# img1=results1[0].plot(font_size=40,pil=True)
# cv.imwrite(('ardmega_out.jpg'), img1)
# img2=results2[0].plot(font_size=15,pil=True)
# cv.imwrite(('arduno_out.jpg'), img2)
# img3=results3[0].plot(font_size=30,pil=True)
# cv.imwrite(('rasppi_out.jpg'), img3)
# cv.waitKey()
# cv.destroyAllWindows()
