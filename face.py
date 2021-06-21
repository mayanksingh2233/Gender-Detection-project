from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import cvlib as cv
import os

#Load_model
model = load_model('gender_detection.model')

#open webcam
webcam = cv2.VideoCapture(0)

classes=['man','women']

#loop through  Frame
while webcam.isOpened():
    #read frame from webcam
    stauts, frame= webcam.read()

    #apply  face detection 
    face, confidence = cv.detect_face(frame)

    #loop through detected faces
    for idx, f in enumerate(face):

        # get coner points of face rectangle
        (startx, starty) =f[0], f[1]
        (endx, endy) = f[2], f[3]

        #draw a rectangle over face
        cv2.rectangle(frame, (startx , starty), (endx , endy),(0,0,255),2)

        #crop the detected face region
        face_crop = np.copy(frame[starty:endy, startx:endx])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue


        #preprocessing the gender detection
        face_crop = cv2.resize(face_crop,(96,96))
        face_crop = face_crop.astype('float32') / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop , axis=0)

        #apply gender detection on  face
        conf = model.predict(face_crop)[0] #model.predict return a 2D matrix 

        #get labels with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = starty - 10  if starty - 10 > 10 else starty + 10

        #write labels and confidence above the faces
        cv2.putText(frame, label, (startx, starty),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    
    #display output
    cv2.imshow('Gender Detection',frame)

    #press "Q" to  stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release sources
webcam.release()
cv2.destroyAllWindows()