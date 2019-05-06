import frstnt as frst
import cv2
import numpy as np

class IrisDetector:
    def rescale_frame(self, frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    def get_face_irises(self, frame, left_eye, right_eye):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        right_candidates = self.eye_iris_frst(frame, right_eye)
        left_candidates = self.eye_iris_frst(frame, left_eye)

        return left_candidates, self.get_mass_center(left_eye), right_candidates, self.get_mass_center(right_eye)


    """
    Returns cropped frame and origin
    """
    def get_eye_frame(self, frame, eye_borders):
        min_x, max_x = np.min(eye_borders[:, 0])-10, np.max(eye_borders[:, 0])+10
        min_y, max_y = np.min(eye_borders[:, 1])-10, np.max(eye_borders[:, 1])+10

        new_borders = map(lambda x: np.substract(x, (min_x, min_y)), eye_borders)

        return frame[min_y:max_y, min_x:max_x], (min_x, min_y)

    """
    Performs frst with the frame to search for
    irises. The radius of the iris is estimated
    using the eye borders.
    """
    def eye_iris_frst(self, frame, eye_borders):
        #  Estimate the iris radius
        eye_width = np.sqrt(np.sum(np.power(eye_borders[0] - eye_borders[3], 2)))
        iris_radius = int(eye_width * 0.42 / 2)

        gray, origin = self.get_eye_frame(frame, eye_borders)

        # cv2.imshow("gray", gray)

        if gray.size == 0:
            return []

        #  Make frst with the frame
        frst_result = frst.frst(gray, iris_radius, 2, 0, 1, "DARK")

        #  Normalize the reuslt
        frst_result = frst_result - np.min(frst_result)
        frst_result = frst_result / np.max(frst_result)
        frst_result = np.uint8(frst_result*255)

        #  Get only important dots
        ret, frst_image = cv2.threshold(frst_result, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        mc = self.get_image_contours(frst_image)

        #  Return to origin coordinates
        mc = map(lambda x: np.add(x, origin), mc)

        #  Filter only thoes that inside the eye
        mc = self.filter_inside_eye(mc, eye_borders)

        #  Get the best candidate
        if len(mc) == 0:
            return []

        best_candidate = mc[0]
        for cand in mc:
            point = np.subtract(cand, origin)
            if point[0] > frst_result.shape[0] or point[1] > frst_result.shape[1]:
                continue
            x0,x1 = point[0]-iris_radius, point[0]+iris_radius
            y0,y1 = point[1]-iris_radius, point[1]+iris_radius

            a0,a1 = best_candidate[0]-iris_radius, best_candidate[0]+iris_radius
            b0,b1 = best_candidate[1]-iris_radius, best_candidate[1]+iris_radius

            if np.mean(frst_result[y0:y1, x0:x1]) < np.mean(frst_result[b0:b1, a0:a1]) or np.isnan(np.mean(frst_result[b0:b1, a0:a1])):
                best_candidate = point

        return [np.add(best_candidate, origin)]

    """
    Filters list of points to have only points that
    are inside the eye borders.
    """
    def filter_inside_eye(self, points, eye_borders):
        return list(filter(lambda x: cv2.pointPolygonTest(eye_borders, tuple(x), False) >= 0, points))

    """
    Returns the mass center of a contour.
    """
    def get_mass_center(self, contour):
        mu = cv2.moments(contour, True)

        if mu['m00'] == 0:
            return [None, None]

        return (int(mu['m10'] / mu['m00']), int(mu['m01']/mu['m00']))

    """
    Return list of center of mass for contours in the
    frame.
    """
    def get_image_contours(self, frame):
        #  Get contours
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #  Get the moments
        mc = []
        for contour in contours:
            center = self.get_mass_center(contour)

            if center is None or center == [None, None]:
                continue

            mc.append(center)

        return mc
