import pickle
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, RNN, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import RandomUniform
import matplotlib.pyplot as plt

def read_file(file_name, column_size):
    #Reading iris data
    df = pd.read_csv(file_name, header=None, sep="\n")
    #Droping first 8 row for reading data
    df = df.iloc[column_size:]
    #Converting into list
    df = df.to_string(header=False,
                  index=False,
                  index_names=False).split("\n")
    
    #Converting list to Dataframe
    train_x = pd.DataFrame([sub.split(",") for sub in df])
    
    #Getting last column as resultant class labels
    train_y = pd.DataFrame(train_x.iloc[:,-1])
    #Droping last colum from train set
    train_x = train_x.drop(train_x.columns[-1], 1)

    #Converting string to float
    train_x = str_to_int(train_x)
    #rename columns
    train_y.columns = [0]
    train_y = str_to_int(train_y)
    #set label range 0 to 2
    train_y = reduceOne(train_y)
    return train_x, train_y

def reduceOne(df):
    col = [i for i in range(0, len(df.columns))]
    df[col] = df[col] -1
    return df

def str_to_int(df):
    col = [i for i in range(0, len(df.columns))]
    df[col] = df[col].astype('float64')
    return df

def data_fusion(df1, df2):
    #Concatenate the two dataset
    fusion_data = pd.concat([df1, df2], axis=1)
    fusion_data = fusion_data.rename(columns={x:y for x,y in zip(fusion_data.columns,range(0,len(fusion_data.columns)))})
    return fusion_data

def preprocess(df):
    #Min-Max normalization applied
    col = [0, 1, 2, 3, 4]
    df[col] = df[col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df

def create_model_1(df, dim):
    #Input Layer
    model = Sequential()
    model.add(Dense(24, activation="relu", input_dim=dim, kernel_regularizer="l1"))
    
    #Hidden Layers
    model.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.13))
    model.add(Dense(48, activation="relu"))
    model.add(Dropout(0.2))
    #model.add(Dense(28, activation="relu"))
    #model.add(Dropout(0.1))

    #Output Layer
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    
    return model

def create_model_2(df, dim):
    #Input Layer
    model = Sequential()
    model.add(Dense(48, activation="relu", input_dim=dim, kernel_initializer="he_uniform", kernel_regularizer="l1_l2"))
    model.add(Dropout(0.15))
    
    #Hidden Layers
    model.add(Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    #Output Layer
    model.add(Flatten())
    model.add(Dense(3, activation="softmax"))
    
    return model

def create_model_3(df, dim):
    #Input Layer
    model = Sequential()
    model.add(Dense(36, activation="relu", input_dim=dim, kernel_regularizer="l1"))
    model.add(Dropout(0.2))
    
    #Hidden Layers
    model.add(Dense(48, activation="relu", kernel_regularizer="l1"))
    model.add(Dropout(0.2))
    model.add(Dense(56, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.1))
    model.add(Dense(36, activation="relu"))
    model.add(Dropout(0.2))
    #Output Layer
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    
    return model

def draw_graph(history):
    #print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def save_processed_data(df, name):
    file = open(name, 'ab') 
      
    # source, destination 
    pickle.dump(df, file)                      
    file.close() 

def load_processed_data(name):
    file = open(name, 'rb')      
    db = pickle.load(file) 

    return db

def main():
    #Loading processed data
    if os.path.exists("./GeoTrainX"):
        geo_train_x = load_processed_data("GeoTrainX")
    if os.path.exists("./GeoTrainY"):
        geo_train_y = load_processed_data("GeoTrainY")
    if os.path.exists("./GeoTestX"):
        geo_test_x = load_processed_data("GeoTestX")
    if os.path.exists("./GeoTestY"):
        geo_test_y = load_processed_data("GeoTestY")
    if os.path.exists("./TextTrainX"):
        texture_train_x = load_processed_data("TextTrainX")
    if os.path.exists("./TextTrainY"):
        texture_train_y = load_processed_data("TextTrainY")
    if os.path.exists("./TextTestX"):
        texture_test_x = load_processed_data("TextTestX")
    if os.path.exists("./TextTestY"):
        texture_test_y = load_processed_data("TextTestY")
    if os.path.exists("./FusionTrainX"):
        fusion_train_x = load_processed_data("FusionTrainX")
    if os.path.exists("./FusionTestX"):
        fusion_test_x = load_processed_data("FusionTestX")
    else:
        #X represent data set itself, Y represent the class labels
        #Geometric dataset
        geo_train_x, geo_train_y = read_file('./IrisGeometicFeatures_TrainingSet.txt', 8)
        geo_test_x, geo_test_y = read_file('./IrisGeometicFeatures_TestingSet.txt', 8)

        #Applying preprocess to dataset
        geo_train_x = preprocess(geo_train_x)
        geo_test_x = preprocess(geo_test_x)

        #Texture dataset
        texture_train_x, texture_train_y = read_file('./IrisTextureFeatures_TrainingSet.txt', 9603)
        texture_test_x, texture_test_y = read_file('./IrisTextureFeatures_TestingSet.txt', 9603)
        
        #Combining two data set
        fusion_train_x = data_fusion(geo_train_x, texture_train_x)
        fusion_test_x = data_fusion(geo_test_x, texture_test_x)

        #Saving geometric data
        save_processed_data(geo_train_x, "GeoTrainX")
        save_processed_data(geo_train_y, "GeoTrainY")
        save_processed_data(geo_test_x, "GeoTestX")
        save_processed_data(geo_test_y, "GeoTestY")

        #Saving texture data
        save_processed_data(texture_train_x, "TextTrainX")
        save_processed_data(texture_train_y, "TextTrainY")
        save_processed_data(texture_test_x, "TextTestX")
        save_processed_data(texture_test_y, "TextTestY")

        #Saving combined data
        save_processed_data(fusion_train_x, "FusionTrainX")
        save_processed_data(fusion_test_x, "FusionTestX")

    #Model
    model = create_model_1(geo_train_x, 5)
    #model = create_model_2(texture_train_x, 9600)
    #model = create_model_2(fusion_train_x, 9605)

    model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=0.00001), metrics=['acc'])
    model.summary()
    #fit the keras model on the dataset
    es = EarlyStopping(monitor='loss', patience=2, mode='min')
    history = model.fit(geo_train_x, geo_train_y, epochs=280, batch_size=13, verbose=1,  callbacks=[es], validation_split=0.2)
    #evaluate the keras model
    _, accuracy = model.evaluate(geo_train_x, geo_train_y)
    print('Accuracy: %.2f' % (accuracy*100))

    #Draw the graph of model on each epoch
    draw_graph(history)

    #Test results
    y_pred = model.predict(geo_test_x)
    print(y_pred.mean())
    print(y_pred[0].mean())
    print(y_pred[1].mean())
    print(y_pred[2].mean())

if __name__ == "__main__":
    main()