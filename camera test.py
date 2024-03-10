import numpy as np
import cv2
import pickle

#Camera parameter setup

frameWidth= 640 # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75 # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
#camera video setup

cap = cv2.VideoCapture(101)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#import trained module

pickle_in=open("model_trained.p","rb") ## rb = READ BYTE
model=pickle.load(pickle_in)
def grayscale(img):
 img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 return img
def equalize(img):
 img =cv2.equalizeHist(img)
 return img
def preprocessing(img):
 img = grayscale(img)
 img = equalize(img)
 img = img/255
 return img


