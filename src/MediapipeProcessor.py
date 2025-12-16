import mediapipe as mp

class MediaPipeProcessor():
    def __init__(self) -> None:
        Holistic = mp.solutions.holistic.Holistic
        self.model = Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_pose_world_landmarks(self, image):
        results = self.model.process(image)    
        return results.pose_world_landmarks