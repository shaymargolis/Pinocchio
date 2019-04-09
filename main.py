import cv2
import numpy as np

from image_source import ImageSource
from sixty_eight import SixtyEightInterpreter

class Analyzer:
    def __init__(self, source, interpreter):
        self.source = source
        self.interpreter = interpreter

    def start(self):
        result = list()

        frame_index = 0

        while not self.source.is_finished():
            #  Get the next frame
            ret, frame = self.source.next_frame()

            #  Only analyze if the frame is not null
            if ret == False:
                continue

            #  Get the face detections
            detections = self.interpreter.interpret(frame)

            #  Show the result
            cv2.imshow("image", frame)

            #  Append to the result array
            result.append((frame_index, detections))

            print(detections[0, :, :])

            frame_index += 1

        return result

source = ImageSource("Videos/roy_amir_lie.MOV")
orig_interpreter = SixtyEightInterpreter()
analyzer = Analyzer(source, orig_interpreter)
analyzer.start()
