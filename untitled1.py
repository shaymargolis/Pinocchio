import cv2
import dlib
import numpy as np

PIXEL_ADDITION_TO_FACE_X = 100
PIXEL_ADDITION_TO_FACE_Y = 100

class ImageSource:
    def __init__(self, video_path):
        self.video_capture = cv2.VideoCapture(video_path)

    def next_frame(self):
        return self.video_capture.read()

    def is_finished(self):
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            return True

        return False

class SixtyEightInterpreter:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector() #Face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

        self.last_margins = None #  [left_x, right_x, top_y, bottom_y]

    def get_cut_frame(self, frame):
        if self.last_margins is None:
            return frame

        left_x = self.last_margins[0]
        right_x = self.last_margins[1]
        top_y = self.last_margins[2]
        bottom_y = self.last_margins[3]

        return frame[top_y:bottom_y, left_x:right_x]

    def update_frame_by_cut(self, frame, cut_frame):
        if self.last_margins is None:
            return

        left_x = self.last_margins[0]
        right_x = self.last_margins[1]
        top_y = self.last_margins[2]
        bottom_y = self.last_margins[3]

        frame[top_y:bottom_y, left_x:right_x] = cut_frame

    def update_result_by_cut(self, result, cut_frame):
        if self.last_margins is None:
            return

        left_x = self.last_margins[0]
        top_y = self.last_margins[2]

        for i in range(len(result)):
            result[i, :, 0] += left_x
            result[i, :, 1] += top_y

    def update_margins(self, frame, result_vector):
        height, width = frame.shape[0], frame.shape[1]

        left_x = max(0, np.min(result_vector[:, 0])-PIXEL_ADDITION_TO_FACE_X)
        right_x = min(width, np.max(result_vector[:, 0])+PIXEL_ADDITION_TO_FACE_X)

        top_y = max(0, np.min(result_vector[:, 1])-PIXEL_ADDITION_TO_FACE_Y)
        bottom_y = min(height, np.max(result_vector[:, 1])+PIXEL_ADDITION_TO_FACE_Y)

        self.last_margins = [left_x, right_x, top_y, bottom_y]

    def interpret(self, frame):
        cut_frame = self.get_cut_frame(frame)

        #  Get the result from OPENCV face detection
        result = self.opencv_face_detection(cut_frame)
        if len(result) == 0:
            result = self.opencv_face_detection(cut_frame)

        result = np.array(result)

        if len(result) == 0:
            return None

        #  Update the frame to by inside 1080 x 1920 px
        self.update_frame_by_cut(frame, cut_frame)
        self.update_result_by_cut(result, cut_frame)

        #  Get the last margins from the frame
        self.update_margins(frame, result[0])

        #  Normalize the data and return the norm vector
        anchor_point = result[0, 8, :]

        result[0, :, :] -= anchor_point
        result[0, :, :] *= -1

        return result

    def opencv_face_detection(self, frame):
        result = list()

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
