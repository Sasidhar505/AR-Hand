import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False , maxhands=2 , detectconf=0.5 , trackconf=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectconf = detectconf
        self.trackconf = trackconf


        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode , self.maxhands , 1 , self.detectconf , self.trackconf)
        self.mpdraw = mp.solutions.drawing_utils


    def findhands(self , img , draw=True):
        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlndmrks in self.results.multi_hand_landmarks:
                if draw :
                    self.mpdraw.draw_landmarks(img , handlndmrks , self.mphands.HAND_CONNECTIONS)
        
        return self.results , img
    

    def findposition(self , img , handno = 0 , draw = True):

        lmlist = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id , lm in enumerate(myhand.landmark):
                h , w , c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                lmlist.append([id , cx , cy ])
                if draw : 
                    cv.circle(img , (cx,cy) , 7 , (255 , 0 , 255) , cv.FILLED)


        return lmlist









def main():
    ptime = 0
    ctime = 0
    cap = cv.VideoCapture(1)
    detector = HandDetector()
    while True:
        success , img = cap.read()
        img = detector.findhands(img)
        landmarks = detector.findposition(img , 0 , False)
        if len(landmarks) != 0 :
            print(landmarks[0])
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv.putText(img , str(int(fps)) , (10,70) , cv.FONT_HERSHEY_PLAIN , 3 , (255,0,0) , 3)

        cv.imshow("Image" , img)
        cv.waitKey(1)
 




if __name__ == "__main__" :
    main()