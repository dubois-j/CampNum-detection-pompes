import cv2
import math

class pompeCount:

    def __init__(self, windowSize=25):
        self.pompeCount = 0
        self.modelOutput = []
        self.window = windowSize


    def findAngle(self, landmarks, p1, p2, p3):   
        #Get the landmarks
        self.landmarks = landmarks.landmark

        x1 = self.landmarks[p1].x
        y1 = self.landmarks[p1].y
        x2 = self.landmarks[p2].x
        y2 = self.landmarks[p2].y
        x3 = self.landmarks[p3].x
        y3 = self.landmarks[p3].y

        #Calculate Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        # make sure angle is between 0 and 180 degrees
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle

        return angle
    
    def isPompe(self, modelResult, thresh, window):
        self.modelOutput.append(modelResult)
        if len(self.modelOutput)< window:
            return 
        elif self.modelOutput[-window] - self.modelOutput[-1] > 0:
            if self.modelOutput[-1] < thresh:
                if self.modelOutput[-2] > thresh:
                    self.pompeCount = self.pompeCount + 1


    def augmentCount(self):
        self.pompeCount = self.pompeCount + 1
        return self.pompeCount
    

    def displayCount(self, image, loc=(50, 100)):
        image = cv2.rectangle(image, (0,0), (350,60), (255,255,255), -1)

        image = cv2.putText(image, "Push-up count:  " + str(self.pompeCount), (15,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (86, 3, 252), 2, cv2.LINE_AA)
        return image


    def getPompeCount(self):
        return self.pompeCount