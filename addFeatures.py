import pandas as pd
import numpy as np
import numpy.linalg as LA
import os
import tqdm

def features_avg(features):
    ind = features.columns.values.copy()
    for i in range(len(ind)):
        ind[i] += "_avg"
    avgs = pd.concat([features.mean()]*features.shape[0], axis = 1)
    avgs = avgs.transpose()
    avgs = pd.DataFrame(np.array(avgs), columns = ind)
    return avgs

def features_std(features):
    ind = features.columns.values.copy()
    for i in range(len(ind)):
        ind[i] += "_std"
    stds = pd.concat([features.std()]*features.shape[0], axis = 1)
    stds = stds.transpose()
    stds = pd.DataFrame(np.array(stds), columns = ind)
    return stds

def features_max(features):
    ind = features.columns.values.copy()
    for i in range(len(ind)):
        ind[i] += "_max"
    maxs = pd.concat([features.max()]*features.shape[0], axis = 1)
    maxs = maxs.transpose()
    maxs = pd.DataFrame(np.array(maxs), columns = ind)
    return maxs

def features_euclidean_dists(features):
    result = pd.DataFrame()


    ind = [str(i) for i in range(7)]
    ind += [str(i) for i in range(8, 68)]
    ind += ["left_eye_", "left_iris_", "right_eye_", "right_iris_"]

    pg = tqdm.tqdm(total = int(len(ind)*(len(ind)-1)/2))

    for i in range(0, len(ind)):
        for j in range(i+1, len(ind)):
            k = ind[i]
            p = ind[j]

            x0, y0, x1, y1 = features[str(k)+"x"], features[str(k)+"y"], features[str(p)+"x"], features[str(p)+"y"]

            result[k+" DIST FROM "+p] = (x0-x1)**2+(y0-y1)**2
            pg.update()

    return result
