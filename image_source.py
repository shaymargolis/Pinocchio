import cv2

class ImageSource:
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)

    def next_frame(self):
        return self.video_capture.read()

    def is_finished(self):
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            return True

        return False
