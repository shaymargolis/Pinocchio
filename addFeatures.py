import pandas as pd
import numpy as np
import numpy.linalg as LA
import os

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

def features_euclidean_dists(features):
    result = pd.DataFrame()

    ind = list(range(1,8))
    ind += list(range(9, 68))

    print("AAAA")
    print(features)

    for i in range(0, len(ind)):
        for j in range(i+1, len(ind)):
            k = ind[i]
            p = ind[j]

            x0, y0, x1, y1 = features[str(k)+"x"], features[str(k)+"y"], features[str(p)+"x"], features[str(p)+"y"]

            result[str(i+1)+"DIST FROM"+str(j+1)] = (x0-x1)**2+(y0-y1)**2
            os.system("cls")
            print(100*(i*len(ind)+j)/(len(ind)**2))

    return result
