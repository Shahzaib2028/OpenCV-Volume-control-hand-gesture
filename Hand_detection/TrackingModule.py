import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode = False, maxHands = 2 , detectionCon = 0.5 , trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands , self.detectionCon , self.trackCon)  # use RGB image so we have to convert it
        self.mpDraw = mp.solutions.drawing_utils


    def FindHands(self , img , draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting here from BRG to RGB
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for i in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, i, self.mpHands.HAND_CONNECTIONS)
        return img

    def FindPositions(self, img , handNo=0 , draw = True):

        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx , cy])
                if draw:
                    if id == 4:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

        return lmList



def main():
    capture = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0
    detector = HandDetector()

    while True:
        success, img = capture.read()
        img = detector.FindHands(img)
        lmList = detector.FindPositions(img)
        if len(lmList) !=0:
            print(lmList[4])
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()