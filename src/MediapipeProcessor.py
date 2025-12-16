import mediapipe as mp
#import PompeCount
import cv2

class MediaPipeProcessor:
    def __init__(self) -> None:
        Holistic = mp.solutions.holistic.Holistic
        self.model = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #self.PompeCount = PompeCount
        self.camera = cv2.VideoCapture(0)

    def get_pose_world_landmarks(self, image):
        results = self.model.process(image)    
        return results.pose_world_landmarks
    
    def show_interface(self):
        print("\nCOUCOU\n")
        while self.camera.isOpened():
            #count_pompe = self.PompeCount.getPompeCount
            _, image = self.camera.read()
            cv2.imshow('MediaPipe', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            
        self.camera.release()
        cv2.destroyAllWindows()
