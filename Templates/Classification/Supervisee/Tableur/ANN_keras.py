from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


xTrain, yTrain = ...,...
xTest, yTest = ...,...

data_to_predict = ...

OPTIMIZER = 'adam'
FUNCTION_LOSS = "mean_squared_error"
NUMBER_NEURONE_HIDDEN_LAYER_ANN = 16

##LAYER
NUMBER_HIDDEN_LAYER = 3


##NEURONE
NUMBER_NEURONE_OUTPUT = 1


##FUNCTION
FUNCTION_ACTIVATION = "relu"
FUNCTION_OUTPOUT = "sigmoid"
BATCH_SIZE = 32

##OTHER
NUMBER_EPOCH = 15
assert xTrain.shape[1] == xTest.shape[1] and yTrain.shape[1] == yTest.shape[1] and xTrain.shape[1] == yTrain.shape[1]
ANN_NUMBER_INPUT = xTrain.shape[1]
METRIC = ["acc"]






def buildANN():

    model = Sequential()
    model.add(Dense(NUMBER_NEURONE_HIDDEN_LAYER_ANN,activation=FUNCTION_ACTIVATION,input_shape=(ANN_NUMBER_INPUT,)))
    for j in range(NUMBER_HIDDEN_LAYER-1):
        model.add(Dense(units=NUMBER_NEURONE_HIDDEN_LAYER_ANN, activation=FUNCTION_ACTIVATION))

    model.add(Dense(NUMBER_NEURONE_OUTPUT,activation=FUNCTION_OUTPOUT))
    model.compile(optimizer=OPTIMIZER,loss=FUNCTION_LOSS,metrics=METRIC)
    return model

ANN = buildANN()

history = ANN.fit(xTrain,
            yTrain,
            epochs=NUMBER_EPOCH,
            batch_size=BATCH_SIZE,
            validation_data=(xTest,yTest),
            verbose=0)

prediction = ANN.predict(data_to_predict)