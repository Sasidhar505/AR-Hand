import cv2 as cv
import HandTracker as ht


width = 1280
height = 720
cap = cv.VideoCapture(1)

detector = ht.HandDetector(maxhands=1 , detectconf=0.8)

while True :
    success , frame = cap.read()
    frame = cv.resize(frame, (width , height))
    # frame = imutils.resize(frame , height=height)

    hands , frame = detector.findhands(frame)
    
    if hands:
        hand=hands[0]
        # landmarks = hand['lmList']
        # print(landmarks)


    cv.imshow('Image' , frame)
    cv.waitKey(1)