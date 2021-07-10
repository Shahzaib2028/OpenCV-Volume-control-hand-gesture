import cv2
import mediapipe as mp
import numpy as np
import time
import TrackingModule as tm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

capture = cv2.VideoCapture(0)
previousTime = 0
currentTime = 0

detect = tm.HandDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success , img = capture.read()
    img = detect.FindHands(img)
    lmList = detect.FindPositions(img , draw=False)
    if len(lmList) !=0 :
        x1 , y1 = lmList[4][1] , lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx , cy = (x1+x2)//2 , (y1+y2)//2
        length_of_line = math.hypot((x2-x1) , (y2-y1))

        cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0) , 3)

        #length of line ranges from 20 - 230
        #vol ranges from -65 - 0

        vol = np.interp(length_of_line,[20,230], [minVol,maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)


    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)