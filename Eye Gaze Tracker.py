import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from sklearn.linear_model import LinearRegression

OutputX = []
OutputY = []
CoordinateX = []
CoordinateY = []

# pyautogui.FAILSAFE = False

Source = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence= 0.5
    )
DisplayWidth, DisplayHeight = pyautogui.size() # Screen Size to scale up/down
RetrainScale = 1
Pause = 0
Calibration = 0     
CalibrationStart = 0
start = 0
end = 0
Retrain = False

XMiddle = []
XTopLeft = []
XTopRight = []
XBottomLeft = []
XBottomRight = []

YMiddle = []
YTopLeft = []
YTopRight = []
YBottomLeft = []
YBottomRight = []


FaceMiddleX = []
FaceMiddleY = []
AverageEyeWidth = []
XRetrainInput = []
YRetrainInput = []

while True:
    _, Frame = Source.read()


    Frame = cv2.flip(Frame, 1) # Mirror Image
    RGBFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(RGBFrame)
    landmark_points = output.multi_face_landmarks
    Frame_h, Frame_w, _ = Frame.shape     

    if Pause == 1:
        print ("Paused")

    elif landmark_points:

        landmarks = landmark_points[0].landmark
        MouthDiff = landmarks[87].y - landmarks[82].y

        Facex = int(landmarks[168].x * Frame_w)
        Facey = int(landmarks[168].y * Frame_h)
        cv2.circle(Frame, (Facex, Facey), 3, (255, 255, 0))

        Rightx = int(landmarks[473].x * Frame_w)
        Righty = int(landmarks[473].y * Frame_h)
        cv2.circle(Frame, (Rightx, Righty), 3, (0, 255, 0))

        Leftx = int(landmarks[468].x * Frame_w)
        Lefty = int(landmarks[468].y * Frame_h)
        cv2.circle(Frame, (Leftx, Lefty), 3, (0, 255, 0))

        RightEyeRightX =  int(landmarks[474].x * Frame_w)
        RightEyeRightY = int(landmarks[474].y * Frame_h)
        RightEyeLeftX =  int(landmarks[476].x * Frame_w)
        RightEyeLeftY = int(landmarks[476].y * Frame_h) 
        RightEyeWidth  = int(((RightEyeRightX - RightEyeLeftX)**2 +(RightEyeRightY - RightEyeLeftY)**2)**0.5)

        LeftEyeRightX =  int(landmarks[469].x * Frame_w)
        LeftEyeRightY = int(landmarks[469].y * Frame_h)
        LeftEyeLeftX =  int(landmarks[471].x * Frame_w)
        LeftEyeLeftY = int(landmarks[471].y * Frame_h) 
        LeftEyeWidth  = int(((LeftEyeRightX - LeftEyeLeftX)**2 +(LeftEyeRightY - LeftEyeLeftY)**2)**0.5)

        EyeWidth = int((LeftEyeWidth + RightEyeWidth)/2)


        if Leftx > 0 and Rightx > 0:
            Midx = int((Rightx + Leftx)/2)
        else:
            Midx = 0
        
        if Lefty > 0 and Righty > 0:
            Midy = int((Righty + Lefty)/2)
        else:
            Midy = 0

        cv2.circle(Frame, (Midx, Midy), 3, (255, 0, 0)) 



        if Calibration == 0:
            Calibrater = np.zeros([DisplayHeight,DisplayWidth,3],dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            Calibrater.fill(255)
            cv2.putText(Calibrater, "Press c to start.", (100, int(DisplayHeight - 350)), font, 2, (0,0,0), 2)
            cv2.putText(Calibrater, "Try to sit in the center of frame.", (100, int(DisplayHeight - 250)), font, 2, (0,0,0), 2)
            cv2.putText(Calibrater, "Stare at red dot untill it turns green", (100, int(DisplayHeight - 150)) , font, 2, (0,0,0), 2)

            cv2.circle(Calibrater, (100, 100), 5, (0, 0, 0), 10)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 0, 0), 10)
            cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 0, 0), 10)
            cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 0, 0), 10)
            cv2.circle(Calibrater, (int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 0, 0), 10)


            if (start == 0):
                end = 0
            else:
                end = time.time()
            if CalibrationStart == 1:
                start = time.time()
                CalibrationStart = 0
            diff = end*1000 - start*1000

            if diff < 2000 and diff > 1:
                cv2.circle(Calibrater, (100, 100), 5, (0, 0, 255), 20)
            elif  diff  > 2000 and diff < 5000:
                #Top Left
                cv2.circle(Calibrater, (100, 100), 5, (0, 0, 255), 20)
                if(diff> 2500):
                    XTopLeft.append(Midx)
                    YTopLeft.append(Midy)
            elif diff > 5000 and diff < 8000:
                #Top Right
                cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 0, 255), 20)
                if(diff> 5500):
                    XTopRight.append(Midx)
                    YTopRight.append(Midy)
            elif diff > 8000 and diff < 11000:
                #Bottom Left
                cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 0, 255), 20)
                if(diff> 8500):
                    XBottomLeft.append(Midx)
                    YBottomLeft.append(Midy)
            elif diff > 11000 and diff < 14000:
                #Bottom Right
                cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 0, 255), 20)
                if(diff> 11500):
                    XBottomRight.append(Midx)  
                    YBottomRight.append(Midy)
            elif diff > 14000 and diff < 17000:
                #Middle
                cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 0, 255), 20)
                if(diff> 14500):
                    XMiddle.append(Midx)  
                    YMiddle.append(Midy)
                    AverageEyeWidth.append(EyeWidth)
                    FaceMiddleX.append(Facex) 
                    FaceMiddleY.append(Facey)
            elif diff > 17000:
                #All Green | Breather before closing
                cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
                cv2.circle(Calibrater, ( int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 255, 0), 20)

            cv2.imshow('Calibration Window', Calibrater)
            if(diff > 18000):
                Calibration = 1
                print ("Calibrated")

                Xinput = np.concatenate((XTopLeft, XTopRight, XBottomLeft, XBottomRight, XMiddle)).reshape(-1, 1)
                Xlist1 = [0]*len(XTopLeft)
                Xlist2 = [DisplayWidth]*len(XTopRight)
                Xlist3 = [0]*len(XBottomLeft)
                Xlist4 = [DisplayWidth]*len(XBottomRight)
                Xlist5 = [DisplayWidth/2]*len(XMiddle)
                Xval = np.concatenate((Xlist1, Xlist2, Xlist3, Xlist4, Xlist5))


                Yinput = np.concatenate((YTopLeft, YTopRight, YBottomLeft, YBottomRight, YMiddle)).reshape(-1, 1)
                Ylist1 = [0]*len(YTopLeft)
                Ylist2 = [0]*len(YTopRight)
                Ylist3 = [DisplayHeight]*len(YBottomLeft)
                Ylist4 = [DisplayHeight]*len(YBottomRight)
                Ylist5 = [DisplayHeight/2]*len(YMiddle)
                Yval = np.concatenate((Ylist1, Ylist2, Ylist3, Ylist4, Ylist5))


                # Create a LinearRegression object
                modelX = LinearRegression()
                modelY = LinearRegression()

                # Fit the model to the data
                modelX.fit(Xinput, Xval)
                modelY.fit(Yinput, Yval)

                XMiddleFace = int(sum(FaceMiddleX)/len(FaceMiddleX))
                YMiddleFace = int(sum(FaceMiddleY)/len(FaceMiddleY))
                EyeDiameter = int(sum(AverageEyeWidth)/len(AverageEyeWidth))
                XMiddleOfEye = int(sum(XMiddle)/len(XMiddle))
                YMiddleOfEye = int(sum(YMiddle)/len(YMiddle))
        elif (Retrain == True):
            # Create a LinearRegression object

            Retrain = False
            XNewInput = Xinput
            for x in XNewInput: 
                NewX = int(XMiddleOfEye + ((1 + (Scale - 1)/2) * (XMiddleOfEye - x)))
                if NewX > Frame_w:
                    NewX = Frame_w
                if NewX  < 0:
                    NewX = 0
                XRetrainInput.append(NewX)

            YNewInput = Yinput
            for y in YNewInput: 
                NewY = int(YMiddleOfEye + ((1 + (Scale - 1)/2)  * (YMiddleOfEye - y)))
                if NewY > Frame_h:
                    NewY = Frame_h
                if NewY  < 0:
                    NewY = 0
                YRetrainInput.append(NewY)
            
            XNewInput = np.array(XRetrainInput).reshape((-1, 1))
            YNewInput = np.array(YRetrainInput).reshape((-1, 1))

            RetrainX = LinearRegression()
            RetrainY = LinearRegression()

            # Fit the model to the data
            RetrainX.fit(XNewInput, Xval)
            RetrainY.fit(YNewInput, Yval)
            RetrainScale = Scale
        else:
            if(MouthDiff < 0.03):
                cv2.circle(Frame, (Rightx, Righty), 3, (0, 255, 0))
                cv2.circle(Frame, (Leftx, Lefty), 3, (0, 255, 0))
                cv2.circle(Frame, (Midx, Midy), 3, (255, 0, 0))
                if Midx > 0 and Midy > 0:
                    EyeDiaDiff =  EyeWidth - EyeDiameter 
                    if((EyeDiaDiff > 0.1 or EyeDiaDiff < -0.1 )):
                        Scale =1 + EyeDiaDiff/EyeDiameter 
                        if(RetrainScale - Scale > 0.1 or RetrainScale- Scale < 0.1):
                            XRetrainInput = []
                            YRetrainInput = []

                            Retrain = True
                            Scaled = True
                    else:
                        Scaled = False
                     # Predict the value of P using the model
                    NoseDiffX = Facex - XMiddleFace
                    NoseDiffY = Facey - YMiddleFace

                    
                    if (Scaled == True and 0 and Retrain == False):
                        print("Using Retrained")
                        NewScreenX = RetrainX.predict([[Midx]])
                        NewScreenY = RetrainY.predict([[Midy]])
                        pyautogui.moveTo(NewScreenX, NewScreenY)
                    elif ( NoseDiffX  != 0 or NoseDiffY != 0 ):
                        screen_x = modelX.predict([[Midx]])
                        screen_y = modelY.predict([[Midy]])
                        screen_x = screen_x - int(NoseDiffX*1.15)
                        screen_y = screen_y - int(NoseDiffY*1.15)
                        if  screen_x < 0:
                            screen_x = 0
                        if screen_x > DisplayWidth:
                            screen_x = DisplayWidth
                        if screen_y < 0:
                            screen_y = 0
                        if screen_y > DisplayHeight:
                            screen_y = DisplayHeight
                        pyautogui.moveTo(screen_x, screen_y)
                    else:
                        cv2.rectangle(Frame, (int(Frame_w/4), int(Frame_h/4)), (int(3*Frame_w/4), int(3*Frame_h/4)), (0,0, 255), 2)
                        print("Out of frame")
                    
                    diffLeft = landmarks[145].y -  landmarks[159].y
                    diffRight = landmarks[384].y -  landmarks[385].y
                    if (diffLeft > 0  and diffLeft < .025 and diffRight > 0.05): 
                        pyautogui.click()                
            else:
                print("Paused")
    else:
        print("Unable to detect face")

    cv2.imshow('Eye Tracking Project', Frame)
    cv2.waitKey(1)
    k = cv2.waitKey(1) & 0xff

    if  k == 99:
        CalibrationStart = 1
        Caibration = 0
        start = 0
        print ("Key press")
    if  k == 27: # esc to quit
        break
    if k == 32:
        print("Spacebar Click")
        if Pause == 0:
            Pause = 1
        else:
            Pause = 0


Source.release() 
cv2.destroyAllWindows()