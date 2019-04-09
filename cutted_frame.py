import numpy as np

class CuttedFrame:
    def __init__(self, frame, last_margins, pixel_addition_x, pixel_addition_y):
        self.frame = frame
        self.last_margins = last_margins
        self.pixel_addition_x = pixel_addition_x
        self.pixel_addition_y = pixel_addition_y

    def get_frame(self):
        return self.frame

    def get_cut_frame(self):
        if self.last_margins is None:
            return self.frame

        left_x = self.last_margins[0]
        right_x = self.last_margins[1]
        top_y = self.last_margins[2]
        bottom_y = self.last_margins[3]

        return self.frame[top_y:bottom_y, left_x:right_x]

    def update_frame_by_cut(self, cut_frame):
        if self.last_margins is None:
            return

        left_x = self.last_margins[0]
        right_x = self.last_margins[1]
        top_y = self.last_margins[2]
        bottom_y = self.last_margins[3]

        self.frame[top_y:bottom_y, left_x:right_x] = cut_frame

    def update_result_by_cut(self, result):
        if self.last_margins is None:
            return

        left_x = self.last_margins[0]
        top_y = self.last_margins[2]

        for i in range(len(result)):
            result[i, :, 0] += left_x
            result[i, :, 1] += top_y

    def get_new_margins(self, result_vector):
        height, width = self.frame.shape[0], self.frame.shape[1]

        left_x = max(0, np.min(result_vector[:, 0])-self.pixel_addition_x)
        right_x = min(width, np.max(result_vector[:, 0])+self.pixel_addition_x)

        top_y = max(0, np.min(result_vector[:, 1])-self.pixel_addition_y)
        bottom_y = min(height, np.max(result_vector[:, 1])+self.pixel_addition_y)

        return [left_x, right_x, top_y, bottom_y]
