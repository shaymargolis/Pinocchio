import pandas as pd
import numpy as np

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
    s = len(features.columns.values)

    print(features)

    for i in range(0, s, 2):
        for j in range(i+1, s, 2):
            if i+1 >= s or j+1 >= s:
                continue
            x0, y0, x1, y1 = features[str(i)], features[str(i+1)], features[str(j)], features[str(j+1)]
            features[str(i)+"DIST FROM"+str(j)] = LA.norm([x0-x1, y0-y1])
            print(100*(i*s+j)/(s**2))

    return features
