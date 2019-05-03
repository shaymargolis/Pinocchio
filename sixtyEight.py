from analyzer import Analyzer
from sixty_eight_features import SixtyEightInterpreter
from image_source import ImageSource
from tqdm import tqdm
import pandas as pd

def get_sixtyeight_feature(video_path, view = False):
    """
    This function return the 68*2 + 10 features for every frame of the input video
    """
    source = ImageSource(video_path)
    pg = tqdm(total = source.get_length())
    anal = Analyzer(source, SixtyEightInterpreter(), view, pg)
    result = anal.start()
    pg.close()

    ind = [str(i+1) + x for i in range(7) for x in ['x','y']]
    ind += [str(i+1) + x for i in range(8, 68) for x in ['x','y']]
    ind += ["left_eye_x", "left_eye_y", "left_iris_x", "left_iris_y", "right_eye_x", "right_eye_y", "right_iris_x", "right_iris_y", "theta", "eye_ar"]

    result = pd.DataFrame(result, columns = ind)

    return result
