import cv2
import numpy as np
import math
from iris_detector import IrisDetector

class Analyzer:
    def __init__(self, source, interpreter, display, pg=None):
        self.source = source
        self.interpreter = interpreter
        self.display = display
        self.pg = pg

    def start(self, start_frame = -np.inf, end_frame = np.inf):
        result = list()

        frame_index = 0
        iris = IrisDetector()

        while (not self.source.is_finished()) and frame_index <= end_frame:
            #  Update ProgressBar
            frame_index += 1
            if self.pg is not None:
                self.pg.update()

            #  Get the next frame
            frame = self.source.next_frame()

            #  ONLY fOR NOW
            if frame is not None:
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                orig = frame.copy()

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
            #  Iris location
            left, left_mc, right, right_mc = iris.get_face_irises(orig, detections[0][35:41], detections[0][41:47])

            left_iris, right_iris, theta = [None, None], [None, None], None

            if len(right) > 0 and len(left) > 0:
                left_iris = left[0]
                right_iris = right[0]

                cv2.circle(frame, tuple(left[0]), 2, (0, 0, 255), -1)
                cv2.circle(frame, tuple(right[0]), 2, (0, 0, 255), -1)

                left_dist = np.subtract(left[0], left_mc)
                right_dist = np.subtract(right[0], right_mc)

                dist = np.mean([left_dist, right_dist], axis=1)

                theta = math.atan(dist[1] / dist[0])

            #  Show the result
            if self.display:
                cv2.imshow("image", frame)

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

            additional = list(left_mc) + list(left_iris) + list(right_mc) + list(right_iris) + [theta] + [np.average([left_c, right_c])]
            # additional = [np.average([left_c, right_c])]

            #  Append to the result array
            final = list(detections[0, :, :].flatten())
            final += additional
            result.append(final)

        return result
