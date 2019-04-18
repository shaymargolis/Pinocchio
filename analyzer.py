import cv2
import numpy as np

class Analyzer:
    def __init__(self, source, interpreter, display, pg=None):
        self.source = source
        self.interpreter = interpreter
        self.display = display
        self.pg = pg

    def start(self, start_frame = 0, end_frame = np.inf):
        result = list()

        frame_index = 0

        while (not self.source.is_finished()) and frame_index <= end_frame:
            #  Update ProgressBar
            if self.pg is not None:
                self.pg.update()

            #  Get the next frame
            ret, frame = self.source.next_frame()

            #  Only analyze if the frame is not null
            if ret == False and frame_index >= start_frame:
                continue

            #  Get the face detections
            detections = self.interpreter.interpret(frame)

            #  Show the result
            if self.display:
                cv2.imshow("image", frame)

            #  Append to the result array
            result.append(detections[0, :, :].flatten())

            frame_index += 1

        return result
