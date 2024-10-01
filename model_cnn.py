import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras.backend as K
import _pickle as pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from keras.initializers import Constant, TruncatedNormal
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import losses

# Load datasets

X = pickle.load(open("data/X_sisfall_svd.p", "rb"))
y = pickle.load(open("data/y_sisfall_svd.p", "rb"))

num_of_classes = 2
signal_rows = 450
signal_columns = 1
num_of_subjects = 23

#Model
for i in range(num_of_subjects):
    test = y.loc[y[1] == i+1]
    test_index = test.index.values

    train = y[~y.index.isin(test_index)]
    train_index = train.index.values

    y_values = y.iloc[:, 0].values

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_values[train_index], y_values[test_index]
    y_train = np.eye(num_of_classes)[y_train - 1]

    # input layer
    input_signal = Input(shape=(signal_rows, 1))

    # define initial parameters
    bias_init = Constant(value=0.0)
    kernel_init = TruncatedNormal(mean=0.0, stddev=0.01, seed=2018)

    # Feature Extractors

    conv11 = Conv1D(16, kernel_size=32, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(input_signal)
    bn11 = BatchNormalization()(conv11)
    actv11 = Activation('relu')(bn11)
    conv12 = Conv1D(32, kernel_size=32, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(actv11)
    bn12 = BatchNormalization()(conv12)
    actv12 = Activation('relu')(bn12)
    flat1 = Flatten()(actv12)

    conv21 = Conv1D(16, kernel_size=42, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(input_signal)
    bn21 = BatchNormalization()(conv21)
    actv21 = Activation('relu')(bn21)
    conv22 = Conv1D(32, kernel_size=42, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(actv21)
    bn22 = BatchNormalization()(conv22)
    actv22 = Activation('relu')(bn22)
    flat2 = Flatten()(actv22)

    conv31 = Conv1D(16, kernel_size=52, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(input_signal)
    bn31 = BatchNormalization()(conv31)
    actv31 = Activation('relu')(bn31)
    conv32 = Conv1D(32, kernel_size=52, strides=1, padding='valid', bias_initializer=bias_init,
                    kernel_initializer=kernel_init)(actv31)
    bn32 = BatchNormalization()(conv32)
    actv32 = Activation('relu')(bn32)
    flat3 = Flatten()(actv32)

    # merge
    merge = concatenate([flat1, flat2, flat3])

    # dropout & fully-connected layer
    dropout = Dropout(0.9, seed=2018)(merge)
    output = Dense(num_of_classes, activation='softmax', bias_initializer=bias_init, kernel_initializer=kernel_init)(dropout)

    # result
    model = Model(inputs=input_signal, outputs=output)

    # summarize layers
    print(model.summary())

    model_dir = 'model/sisfall_svd_conv1d_10/' + str(i+1) + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = model_dir + 'weights.{epoch:02d}-{accuracy:.2f}.hdf5'
    cp_cb = ModelCheckpoint(model_file, monitor='accuracy', verbose=1, save_best_only=True, mode='auto', save_freq=1)

    adam = Adam(lr=0.0001)
    model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    model.fit(np.expand_dims(X_train, axis=2), y_train, batch_size=32, epochs=50, verbose=2, validation_split=0.10,
              callbacks=[cp_cb], shuffle=True)

    del model
    K.clear_session()


# Cross Validation

acc = []

for i in range(num_of_subjects):
    test = y.loc[y[1] == i+1]
    test_index = test.index.values

    train = y[~y.index.isin(test_index)]
    train_index = train.index.values

    y_values = y.iloc[:, 0].values

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_values[train_index] - 1, y_values[test_index] - 1

    print ("\n-----------------------", str(i+1), "-fold -----------------------")
    path_str = 'model/sisfall_svd_conv1d_10/' + str(i+1) + '/'
    for path, dirs, files in os.walk(path_str):
        dirs.sort()
        files.sort()
        top_acc = []
        top_acc.append(files[-1])
        files = top_acc
        for file in files:
            print ("========================================")
            print (os.path.join(path, file))
            model = load_model(os.path.join(path, file))
            
            pred = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
            print ("------ TRAIN ACCURACY: ", file, " ------")
            print (accuracy_score(y_train, np.argmax(pred, axis=1)))
            print (confusion_matrix(y_train, np.argmax(pred, axis=1)))

            pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
            print ("------ TEST ACCURACY: ", file, " ------")
            print (accuracy_score(y_test, np.argmax(pred, axis=1)))
            print (confusion_matrix(y_test, np.argmax(pred, axis=1)))

            del model
            K.clear_session()

    acc.append(accuracy_score(y_test, np.argmax(pred, axis=1)))

print (acc)
print (np.mean(acc))
print (np.std(acc))