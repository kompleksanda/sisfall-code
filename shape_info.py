import pandas as pd
import _pickle as pickle



X = pickle.load(open("data/X_sisfall_svd.p", "rb"))
y = pickle.load(open("data/y_sisfall_svd.p", "rb"))
y = pd.DataFrame(y)

print ("X_sisfall:", X.shape)
print ("y_sisfall:", y.shape)


