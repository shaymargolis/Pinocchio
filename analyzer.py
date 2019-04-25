import cv2
import numpy as np

class Analyzer:
    def __init__(self, source, interpreter, display, pg=None):
        self.source = source
        self.interpreter = interpreter
        self.display = display
        self.pg = pg

    def start(self, start_frame = -np.inf, end_frame = np.inf):
        result = list()

        frame_index = 0

        while (not self.source.is_finished()) and frame_index <= end_frame:
            #  Update ProgressBar
            frame_index += 1
            if self.pg is not None:
                self.pg.update()

            #  Get the next frame
            frame = self.source.next_frame()

            #  Only analyze if the frame is not null
            if frame_index <= start_frame:
                continue

            #  Get the face detections
            detections = self.interpreter.interpret(frame)

            #  Show the result
            if self.display:
                cv2.imshow("image", frame)

            # If no face has been detected
            if detections is None:
                continue

            #  Get additional features
            #  Left Eye
            left_a = np.sum(np.power(detections[0, 35] - detections[0, 38], 2))
            left_b = np.sum(np.power(detections[0, 36] - detections[0, 40], 2))
            left_b += np.sum(np.power(detections[0, 37] - detections[0, 39], 2))
            left_c = left_b/(2*left_a)

            #  Right eye
            right_a = np.sum(np.power(detections[0, 41] - detections[0, 44], 2))
            right_b = np.sum(np.power(detections[0, 42] - detections[0, 46], 2))
            right_b += np.sum(np.power(detections[0, 43] - detections[0, 45], 2))
            right_c = right_b/(2*right_a)

            additional = [np.average([left_c, right_c])]

            #  Append to the result array
            final = list(detections[0, :, :].flatten())
            final += additional
            result.append(final)



        return result
