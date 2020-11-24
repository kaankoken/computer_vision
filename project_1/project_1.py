import glob
import cv2
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt 
import itertools

def display_images(img, label: str):
    #Display method for images
    cv2.imshow(label, mat=img)
    
    if cv2.waitKey(0) and 0xff == 27:
        cv2.destroyAllWindows()

def create_labels(data):
    #create label list for given dataset
    #Ex: label 0 -> airplanes
    labels = []
    for i in range(len(data)):
        arr = [i] * len(data[i])
        labels.extend(arr)
    return labels

def knn_model(k_value, train_set, test_set, t1, t2, level):
    #getting labels of training set
    training_label = create_labels(t1)
    test_label = create_labels(t2)

    #Creating the model
    knn = KNeighborsClassifier(n_neighbors=k_value, metric="euclidean", algorithm="brute", n_jobs=-1)
    
    if level != 1:
        #reshaping the train dataset without losing information
        nsamples, nx, ny, nz = train_set.shape
        d2_train_dataset = train_set.reshape((nsamples,nx*ny*nz))

        #reshaping the test dataset without losing information
        nsamples, nx, ny, nz = test_set.shape
        d2_test_dataset = test_set.reshape((nsamples,nx*ny*nz))

        #train the model
        knn.fit(d2_train_dataset, training_label)
    else:
        knn.fit(train_set, training_label)
    index = 0
    class_labels = ["airplanes", "bonsai", "chair", "ewer", "faces", "flamingo",
                    "guitar", "leopards", "motorbikes", "starfish"]
    
    #displays the prediction
    #Doing predicition for each class individually
    mean = 0.0
    for i, j in zip(t2, class_labels):
        if level != 1:
            pred = knn.predict(d2_test_dataset[index: index + len(i)])
        else:
            pred = knn.predict(test_set[index: index + len(i)])
        print("Class: {}, Accuracy of Prediction: {}".format(j, metrics.accuracy_score(test_label[index: index + len(i)], pred)))
        mean += metrics.accuracy_score(test_label[index: index + len(i)], pred)
        index = len(i)
    print("")
    print("Average of prediciton: {}".format(mean/10))

def divided_images_with_histogram(data, level, isRGB, quantization):
    #Create the new dataset using image that divided into grids
    images = []
    for i in data: 
        for k in i:
            images.append(divide_image_v2(k, level, isRGB, quantization))

    images = np.array(images) 
   
    return images

def divide_image(img, level: int, rgbStatus: bool, quantization: int):
    #diving image into 4 equal pieces -> 2x2
    tiles = []

    if level == 2:
        M = int(np.round(img.shape[0] / 2))
        N = int(np.round(img.shape[1] / 2))

        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

    else:  #diving image into 16 equal pieces -> 4x4
        M = int(np.round(img.shape[0] / 4))
        N = int(np.round(img.shape[1] / 4))

        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
    
    image = None
    #Checks whether image BGR or not
    if not rgbStatus:
        ax = [plt.hist(i.ravel(), bins = quantization)[0] for i in tiles]
        #Gray image histogram values
        image = ax
    else:
        #BGR image histogram values
        ax_b = [plt.hist(i[:, :, 0].ravel(), bins = quantization)[0] for i in tiles]
        ax_g = [plt.hist(i[:, :, 1].ravel(), bins = quantization)[0] for i in tiles]
        ax_r = [plt.hist(i[:, :, 2].ravel(), bins = quantization)[0] for i in tiles]

        #combine_channels
        image = combine_channel(ax_b, ax_g, ax_r)
    #combine all histogram values as one 
    hist = 0
    for i in image:
        hist += i
    return hist            

def divide_image_v2(img, level: int, rgbStatus: bool, quantization: int):
    #diving image into 4 equal pieces -> 2x2
    tiles = []

    if level == 2:
        M = int(np.round(img.shape[0] / 2))
        N = int(np.round(img.shape[1] / 2))

        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

    else:  #diving image into 16 equal pieces -> 4x4
        M = int(np.round(img.shape[0] / 4))
        N = int(np.round(img.shape[1] / 4))

        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

    hist_values = []
    #Checks whether image BGR or not
    if not rgbStatus:
        ax = [plt.hist(i.ravel(), bins = quantization)[0] for i in tiles]
        #Gray image histogram values
        hist_values.extend(ax)
    else:
        #Individual BGR channels
        for i in tiles:
            ax_b = plt.hist(i[:, :, 0].ravel(), bins = quantization)[0]
            ax_g = plt.hist(i[:, :, 1].ravel(), bins = quantization)[0]
            ax_r = plt.hist(i[:, :, 2].ravel(), bins = quantization)[0]

            #combination of 3 channel
            comb_img = list(itertools.product(*[ax_b, ax_g, ax_r]))

            #sum of the 3 channels
            comb_img = combine_channel(comb_img)
            
            #transpose of the image
            std_image = np.transpose(comb_img)
            #burada bişiler olması lazım
            
            hist_values.append(std_image)
    hist_values = normalize(hist_values, norm='l1')
    #concatanate the image asx 2x2 array
    matrix = np.array([[hist_values[0], hist_values[1]], [hist_values[1], hist_values[2]]])

    return matrix
               
def combine_channel(img):
    #Combine BGR channels as one
    combined = [i[0] + i[1] + i[2] for i in img]
    return combined

def histogram(rgbStatus:bool, quantization: int, image):
    hist_values = []

    for label in image:
        if not rgbStatus:
            ax = [plt.hist(i.ravel(), bins = quantization)[0] for i in label]
            #put values between 0 to 1
            std_image = normalize(ax, norm='l1')
            
            hist_values.extend(std_image)
        else:
            #Individual BGR channels
            ax_b = [plt.hist(i[:, :, 0].ravel(), bins = quantization)[0] for i in label]
            ax_g = [plt.hist(i[:, :, 1].ravel(), bins = quantization)[0] for i in label]
            ax_r = [plt.hist(i[:, :, 2].ravel(), bins = quantization)[0] for i in label]

            #combine_channels
            std_image = combine_channel(ax_b, ax_g, ax_r)
            
            #Standardized histogram values for each channel
            #put values between 0 to 1
            std_image = normalize(std_image, norm='l1')
            
            hist_values.extend(std_image)

    return hist_values

def histogram_v2(rgbStatus: bool, quantization: int, images):
    hist_values = []
    for label in images:
        if not rgbStatus:
            ax = [plt.hist(i.ravel(), bins = quantization)[0] for i in label]
            
            hist_values.extend(ax)
        else:
            #Individual BGR channels
            for i in label:
                ax_b = plt.hist(i[:, :, 0].ravel(), bins = quantization)[0]
                ax_g = plt.hist(i[:, :, 1].ravel(), bins = quantization)[0]
                ax_r = plt.hist(i[:, :, 2].ravel(), bins = quantization)[0]

                #combination of 3 channel
                comb_img = list(itertools.product(*[ax_b, ax_g, ax_r]))

                #sum of the 3 channels
                comb_img = combine_channel(comb_img)
                
                #transpose of the image
                std_image = np.transpose(comb_img)
                #burada bişiler olması lazım
                
                hist_values.append(std_image)

    #Standardized histogram values for each channel
    #put values between 0 to 1
    std_image = normalize(hist_values, norm='l1')
    hist_values = pd.DataFrame(std_image)

    return hist_values

def bgr_2_gray(img):
    #Converts all BGR image to Gray scale
    images = []

    for label in img:
        images.append([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in label])

    return images

def read_images(folder_name: str):
    #Reading all images, and creates array of readed images as BGR
    images = []
    class_labels = ["airplanes", "bonsai", "chair", "ewer", "faces", "flamingo",
                    "guitar", "leopards", "motorbikes", "starfish"]
    
    for label in class_labels: 
        images.append([cv2.imread(file) for file in glob.glob(folder_name +"/" + label + "/*.jpg")])
        
    return images

def main():
    #reading BGR images
    training_set = read_images("TrainingSet")
    test_set = read_images("TestSet")
    validation_set = read_images("ValidationSet")

    #converting BGR image to gray scale
    gray_train_set = bgr_2_gray(training_set)
    gray_test_set = bgr_2_gray(test_set)
    gray_validation_set = bgr_2_gray(validation_set)
    
    #quantization = [256, 128, 64]
    df_train = []
    df_test = []
    
    #Level 2 gray image dataset
    quantization = [8, 16, 32]
    #for i in quantization:
    #    level_2_gray_train = divided_images_with_histogram(gray_train_set, 2, False, i)
    #    df_train.append(level_2_gray_train)
    #    level_2_gray_test = divided_images_with_histogram(gray_test_set, 2, False, i)
    #    df_test.append(level_2_gray_test)

    #Level 3 gray image dataset
    #for i in quantization:
    #    level_3_gray_train = divided_images_with_histogram(gray_train_set, 3, False, i)
    #    level_3_gray_test = divided_images_with_histogram(gray_test_set, 3, False, i)
    #    df_train.append(level_3_gray_train)
    #    df_test.append(level_3_gray_test)
    
    #Level 2 BGR image dataset
    #for i in quantization:
    #    level_2_bgr_train = divided_images_with_histogram(training_set, 2, True, i)
    #    df_train.append(level_2_bgr_train)
    #    level_2_bgr_test = divided_images_with_histogram(test_set, 2, True, i)
    #    df_test.append(level_2_bgr_test)

    #Level 3 BGR image dataset
    #for i in quantization:
    #    level_3_bgr_train = divided_images_with_histogram(training_set, 3, True, i)
    #    df_train.append(level_3_bgr_train)
    #    level_3_bgr_test = divided_images_with_histogram(test_set, 3, True, i)
    #    df_test.append(level_3_bgr_test)
    
    #Level 1
    #Standardized histogram values for BGR images
    #for i in quantization:
    #    std_train_bgr = histogram_v2(True, i, training_set)
    #    df_train.append(pd.DataFrame(std_train_bgr))

    #    std_test_bgr = histogram_v2(True, i, test_set)
    #    df_test.append(pd.DataFrame(std_test_bgr))

    #Level 1
    #Standardized histogram values for gray images
    #for i in quantization:
    #    std_train_gray = histogram_v2(False, i, gray_train_set)
    #    df_train.append(pd.DataFrame(std_train_gray))

    #    std_test_gray = histogram_v2(False, i, gray_test_set)
    #    df_test.append(pd.DataFrame(std_test_gray))

    #k_values = [1, 5, 10]

    #for i in range(0,3):
    #    print("Quantization: {}".format(quantization[i]), end=" ")
    #    for k in k_values:
    #        print("K value: {}".format(k))
    #        knn_model(k, df_train[i], df_test[i], training_set, test_set, 1)
    
    #Best Setup
    #Level 1 Gray Level Histogram
    #K = 5
    #Quantizaton 64
    std_train_gray = histogram_v2(False, 64, gray_train_set)
    valid =  histogram_v2(False, 64, gray_validation_set)
    knn_model(5, std_train_gray, valid, gray_train_set, gray_validation_set, 1)
    std_test_gray = histogram_v2(False, 64, gray_test_set)
    knn_model(5, std_train_gray, std_test_gray, gray_train_set, gray_test_set, 1)



if __name__ == "__main__":
    main()