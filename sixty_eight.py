import cv2
import dlib
from cutted_frame import CuttedFrame
import numpy as np


PIXEL_ADDITION_TO_FACE_X = 100
PIXEL_ADDITION_TO_FACE_Y = 100

class SixtyEightInterpreter:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector() #Face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

        self.last_margins = None  # [left_x, right_x, top_y, bottom_y]

    def interpret(self, frame):
        result = self.get_opencv_result(frame)

        # If no face has been detected
        if result is None:
            return None

        #  Normalize the data and return the norm vector
        anchor_point = result[0, 8, :]

        result[0, :, :] -= anchor_point
        result[0, :, :] *= -1

        return result

    def get_opencv_result(self, frame):
        frame_master = CuttedFrame(frame, self.last_margins, PIXEL_ADDITION_TO_FACE_X, PIXEL_ADDITION_TO_FACE_Y)

        #  Get the cutted frame
        cut_frame = frame_master.get_cut_frame()

        used_cut, result = self.fastest_result_possible(frame, cut_frame)

        # If no face has been detected
        if result is None:
            return None

        #  Only if we used the cut_frame (not fallback to
        #  the big original frame)
        #  Update the frame to by inside 1080 x 1920 px
        if used_cut:
            frame_master.update_frame_by_cut(cut_frame)
            frame_master.update_result_by_cut(result)

        #  Get the last margins from the frame
        self.last_margins = frame_master.get_new_margins(result[0])

        return result

    def fastest_result_possible(self, frame, cut_frame):
        used_cut = True
        result = self.opencv_face_detection(cut_frame)
        if result is None or len(result) == 0:
            used_cut = False
            result = self.opencv_face_detection(frame)

        result = np.array(result)

        if len(result) == 0:
            return used_cut, None

        return used_cut, result

    def opencv_face_detection(self, frame):
        result = list()

        if frame is None or frame.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = self.detector(clahe_image, 1) #Detect the faces in the image
        for k,d in enumerate(detections): #For each detected face
            detected_face = list()

            shape = self.predictor(clahe_image, d) #Get coordinates
            for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame

                detected_face.append(np.array([shape.part(i).x, shape.part(i).y]))

            result.append(detected_face)

        return result
