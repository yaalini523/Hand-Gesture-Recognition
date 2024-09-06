import cv2
import torch
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path='Y:/project/miniproject/best.pt'
model=torch.hub.load('ultralytics/yolov5','custom',path,force_reload=True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1000,500))
    results=model(frame)
    frame=np.squeeze(results.render())
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()