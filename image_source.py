import cv2

class ImageSource:
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)

        self.is_finished_bool = False

    def next_frame(self):
        ret, frame = self.video_capture.read()
        if ret == False:
            self.is_finished_bool = True

        return ret, frame

    def is_finished(self):
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            return True

        return self.is_finished_bool

    def get_length(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
