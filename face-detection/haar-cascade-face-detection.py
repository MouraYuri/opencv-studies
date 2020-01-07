import cv2
import numpy as np 

'''

Script to detect faces using webcam


'''

cap = cv2.VideoCapture(0)


#pre-trained model
cc = cv2.CascadeClassifier('./face-detection/haarcascade_frontalface_alt2.xml')

while (True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = cc.detectMultiScale(gray)

    for x, y, w, h in detections:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('frame', gray)
    
    #press esc to close
    key = cv2.waitKey(20) & 0xff
    if (key == 27):
        break


cap.release()
cv2.destroyAllWindows()