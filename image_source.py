import cv2

class ImageSource:
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)
        self.next = self.__raw_next_frame()

        self.is_finished_bool = False

    def next_frame(self):
        frame = self.next

        self.next = self.__raw_next_frame()

        #frame = self.rescale_frame(frame, percent=70)

        return frame

    def __raw_next_frame(self):
        ret, frame = self.video_capture.read()
        if ret == False or frame is None:
            self.is_finished_bool = True

        return frame

    def is_finished(self):
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            return True

        return self.is_finished_bool

    def get_length(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))-1

    def get_fps(self):
        return self.video_capture.get(cv2.CAP_PROP_FPS)

    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
