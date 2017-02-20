import numpy as np
import pandas as pd
import string
import json
import os
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime
from sklearn import preprocessing
from sklearn.externals import joblib
import sys

MODEL_FILENAME = 'model.json'
WEIGHTS_FILENAME = 'model.hdf5'
INTERVALS_FILENAME = 'intervals.csv'
STAT_FILENAME = 'stat.csv'
SCALE_FILENAME = 'scale.pkl'
USERS_FOLDER = 'users'

# NOISE = '0.1'
CLONESIZE = 200
RANDOMYSIZE = 200
RANDOMNOISE = 200
CLONENOISE = 50
EPOCHS = 100
BATCH = 10
VALSPLIT = 0.33

def get_filepathAbs(fileName):
    dir = os.path.dirname(__file__)
    filePath = os.path.join(dir, fileName)
    return filePath

def get_XY(file=None):
    data = pd.read_csv(file)
    # remove first column-which is label
    file_input = data.iloc[:, 1:].values
    file_input = file_input.astype(np.float)
    file_input_size = file_input.shape[1]

    file_output = data[[0]].values
    file_output= file_output.astype(np.float)

    return file_input, file_output, file_input_size

def load_model(user):
    userModelFileRel = USERS_FOLDER + '/' + user + '/' + MODEL_FILENAME
    userWeightsFileRel = USERS_FOLDER + '/' + user + '/' + WEIGHTS_FILENAME
    userScaleFileRel = USERS_FOLDER + '/' + user + '/' + SCALE_FILENAME
    userModelFileAbs = get_filepathAbs(userModelFileRel)
    userWeightsFileAbs = get_filepathAbs(userWeightsFileRel)
    userScaleFileAbs = get_filepathAbs(userScaleFileRel)

    model = model_from_json(open(userModelFileAbs).read())
    model.load_weights(userWeightsFileAbs)

    scale = joblib.load(userScaleFileAbs)

    return model, scale

def save_model(model,scale,user,newUser):
    userModelFileRel = USERS_FOLDER + '/' + user + '/' + MODEL_FILENAME
    userWeightsFileRel = USERS_FOLDER + '/' + user + '/' + WEIGHTS_FILENAME
    userScaleFileRel = USERS_FOLDER + '/' + user + '/' + SCALE_FILENAME
    userModelFileAbs = get_filepathAbs(userModelFileRel)
    userWeightsFileAbs = get_filepathAbs(userWeightsFileRel)
    userScaleFileAbs = get_filepathAbs(userScaleFileRel)

    json_string = model.to_json()
    if newUser:
        os.makedirs(os.path.dirname(userModelFileAbs), exist_ok=True)
        os.makedirs(os.path.dirname(userWeightsFileAbs), exist_ok=True)
        open(userModelFileAbs, 'w').write(json_string)
        model.save_weights(userWeightsFileAbs, overwrite=True)
    else:
        open(userModelFileAbs, 'w').write(json_string)
        model.save_weights(userWeightsFileAbs, overwrite=True)

    # save scale
    joblib.dump(scale, userScaleFileAbs)

def singleTrain(intervals,user,newUser):
    i = json.loads(intervals)
    array = np.array(i)
    length = len(array)
    # X_Train = np.reshape(array,(-1,length))
    # Y_Train = np.array([1])

    X_Train = cloneWithNoise(intervals)
    Y_Train = np.repeat([1],CLONESIZE, axis=0)

    X0_Train = randomX0(intervals)
    Y0_Train = np.repeat([0],RANDOMYSIZE, axis=0)

    X_Train = np.append(X_Train,X0_Train,axis=0)
    Y_Train = np.append(Y_Train, Y0_Train, axis=0)

    # scale X_Train
    scale = preprocessing.StandardScaler().fit(X_Train)
    X_Train_scaled = scale.transform(X_Train)

    if newUser:
        #create new model
        model = Sequential()
        model.add(Dense(length, input_dim=length, init='normal', activation='relu'))
        model.add(Dense(length, input_dim=length, init='normal', activation='relu'))
        model.add(Dense(1, init='normal', activation='sigmoid'))
    else:
        # load existing model
        model = load_model(user)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', #'rmsprop',  # adam
                  metrics=['accuracy'])
    history = model.fit(X_Train_scaled,Y_Train,validation_split=VALSPLIT,batch_size=BATCH,nb_epoch=EPOCHS,verbose=True,shuffle=True)

    #save the model
    save_model(model,scale,user,newUser)

def singleTest(intervals,user):

    model,scale = load_model(user)

    i = json.loads(intervals)
    X_test = np.array(i).reshape(1, -1)
    X_test_scaled = scale.transform(X_test)
    # print(X_test_scaled)
    # length = len(array)
    # X_test = np.reshape(array,(-1,length))

    try:
        classes = model.predict_classes(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)
        print(prob,classes)
        return classes, prob

    except ValueError:
        pass



# def testFromFile():
#     model = load_model()
#     X_test, y_test, X_train_size = get_XY(TESTING_FILE)
#     classes = model.predict_classes(X_test, batch_size=32)
#     print(classes)

def saveIntervals(intervals,Y,user,newUser):
    userIntervalsFileRel = USERS_FOLDER + '/' + user + '/' + INTERVALS_FILENAME
    userIntervalsFileAbs = get_filepathAbs(userIntervalsFileRel)

    i = json.loads(intervals)
    array = np.array(i)
    length = len(array)
    X = np.reshape(array,(-1,length))
    Xs = np.array_str(X)

    if newUser:
        os.makedirs(os.path.dirname(userIntervalsFileAbs), exist_ok=True)
        # Write as a CSV file with headers on first line
        with open(userIntervalsFileAbs, 'wb') as fp:
            #fp.write(','.join(array.dtype.names) + '\n')
            np.savetxt(fp, X,delimiter=",")
    else:
        with open(userIntervalsFileAbs, 'ab') as fp:
            np.savetxt(fp, X,delimiter=",")

def cloneWithNoise(intervals):
    i = json.loads(intervals)
    array = np.array(i)
    length = len(array)
    X = np.reshape(array,(-1,length))
    Xn = np.repeat(X,CLONESIZE,axis=0)
    #add noise
    noise = np.random.normal(0,CLONENOISE,length)

    it = np.nditer(Xn,op_flags=['readwrite'],flags=['multi_index'])
    while not it.finished:
        it[0] = it[0] + noise[it.multi_index[1]]
        if it.multi_index[1] == (length - 1):
            noise = np.random.normal(0, CLONENOISE, length)
        it.iternext()

    return Xn

def randomX0(intervals):
    i = json.loads(intervals)
    array = np.array(i)
    length = len(array)
    X = np.reshape(array,(-1,length))
    Xn = np.repeat(X,RANDOMYSIZE,axis=0)
    #add huge noise
    noise = np.random.normal(0,RANDOMNOISE,length)

    it = np.nditer(Xn,op_flags=['readwrite'],flags=['multi_index'])
    while not it.finished:
        it[0] = it[0] + noise[it.multi_index[1]]
        if it.multi_index[1] == (length - 1):
            noise = np.random.normal(0, RANDOMNOISE, length)
        it.iternext()

    return Xn

def save_stat(user,intervals,Y,prob,newUser):
    # pass
    userStatFileRel = USERS_FOLDER + '/' + user + '/' + STAT_FILENAME
    userStatFileAbs = get_filepathAbs(userStatFileRel)

    # json.JSONEncoder.default = lambda self, obj: (obj.isoformat() if isinstance(obj, datetime.datetime) else None)

    data= {'seq': datetime.now().isoformat(),
           'intervals':intervals,
           'class':np.array_str(Y[0]),
           'prob':np.array_str(prob[0,0]),
           'index':1}

    a = []
    if not os.path.exists(userStatFileAbs):
        os.makedirs(os.path.dirname(userStatFileAbs), exist_ok=True)
        a.append(data)
        with open(userStatFileAbs, 'w') as outfile:
            outfile.write(json.dumps(a, indent=2))
            # json.dump([],outfile)
            # json.dump(data, outfile)
    else:
        with open(userStatFileAbs) as data_file:
            fileJson = json.loads(data_file.read())
            index = len(fileJson)
            data['index'] = index
            fileJson.append(data)
            with open(userStatFileAbs, mode='w') as f:
                f.write(json.dumps(fileJson, indent=2))



def get_stat(user):
    # pass
    userStatFileRel = USERS_FOLDER + '/' + user + '/' + STAT_FILENAME
    userStatFileAbs = get_filepathAbs(userStatFileRel)

    if os.path.exists(userStatFileAbs):
        with open(userStatFileAbs) as data_file:
            data = json.loads(data_file.read())
            return data
    else:
        data = []
        return data

