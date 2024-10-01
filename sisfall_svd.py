import pandas as pd
import os
import re
import numpy as np
import _pickle as pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

data_dir = "data/"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset_path = 'visualizer/SisFall_dataset/'

window_size = 450
window_begin = 100
window_end = 550

labels = []
X = []
for path, dirs, files in os.walk(dataset_path):
    print ("-------")
    dirs.sort()
    files.sort()
    for file in files:
        if "SA" in file:
            label_a = 0
            label_s = 0
            print (file)
            tag = file.split("_")
            label_a = int(re.findall('\d+', tag[0])[0])
            label_s = int(re.findall('\d+', tag[1])[0])
            if 'F' in tag[0]:
                label_a += 19

            file_str = os.path.join(path, file)
            df = pd.read_csv(file_str, header=None)
            stop = int(df.shape[0] / 4)
            idx = []
            for k in range(stop):
                idx.append(k * 4)
            df_converted = df.iloc[idx, 0:3] * 0.00390625

            scaler = StandardScaler()
            svd = TruncatedSVD(n_components=1, random_state=2018)

            if label_a in [1, 2, 3, 4]:
                for j in range(5):
                    scaled = scaler.fit_transform(df_converted[(window_begin + j*window_size) : (window_end + j*window_size)].values)
                    X.append(svd.fit_transform(scaled).reshape((window_size,)))
                    labels.append([label_a, label_s])
            else:
                scaled = scaler.fit_transform(df_converted[window_begin:window_end].values)
                X.append(svd.fit_transform(scaled).reshape((window_size,)))
                labels.append([label_a, label_s])


X = np.vstack(X)
df_label = pd.DataFrame(labels)

pickle.dump(X, open("data/X_sisfall_svd.p","wb"))
pickle.dump(df_label, open("data/y_sisfall_svd.p","wb"))
