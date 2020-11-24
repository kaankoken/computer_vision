import glob
import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt 
import os

def saveAsImage(name, img):
    path = os.getcwd() + "/result"
    if (not os.path.isdir('./result')):
        #creating fodler
        try:
            os.mkdir(os.getcwd() + "/result")
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
    #saving as images
    for i in range(len(img)):
        filename = path + "/" + name + "_" + str(i) + ".bmp"
        cv2.imwrite(filename, img[i])

#Display method for images
def displayImages(img):
    #iterates all the images
    count: int = 1
    for i in img:
        cv2.imshow(str(count), mat=i)
        count+=1
    if cv2.waitKey(0) and 0xff == 27:
        cv2.destroyAllWindows() 

#draw circle to images
def drawCircle(img, circles):
    #if drawing is not possible gives error
    if circles is None:
        print("No circles found")
    else:
        #converts radius and x, y coordinates to 
        #integer values
        circles = np.round(circles[0, :]).astype("int")

        #draw the circles
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (255, 255, 255), 4)

    return img

def getPupil(img):
    images = []
    #iterates list of image
    for i in img:
        #creates copy of original image
        copyImage = copy.deepcopy(i)
        
        #applying threshold
        retval, thresholded = cv2.threshold(i, 30, 255, cv2.THRESH_BINARY)
        
        #creates morphological mask    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        #first dilate than erode the image
        newImg = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
        
        #applying blur
        newImg = cv2.GaussianBlur(newImg, (13, 13), 0)
       
        #thresholding again
        retval, thresholded = cv2.threshold(newImg, 30, 255, cv2.THRESH_BINARY)
       
        #again first dilate than erode the image
        newImg = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
        
        #detecting edges
        newImg = cv2.Canny(newImg, threshold1=retval/2, threshold2=retval)

        #creating circles using hough transformation method
        circles = cv2.HoughCircles(newImg, cv2.HOUGH_GRADIENT, 1, 50, 
            param1=30, param2=15, minRadius=10, maxRadius=70)

        #calling drawing circle method
        #and appeding to new image list
        images.append(drawCircle(copyImage, circles))

    return images

def getIris(img):
    images = []
    for i in img:
        #creates copy of original image
        copyImage = copy.deepcopy(i)

        #creates morphological mask 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        
        #applying threshold
        retval, thresholded = cv2.threshold(i, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        #first dilate than erode the image
        newImg1 = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)

        #detect edges
        newImg = cv2.Canny(newImg1, 5, 70, 3)

        #apply some blurs
        newImg =  cv2.GaussianBlur(newImg, (7, 7), 1)

        #creating circles using hough transformation method
        circles = cv2.HoughCircles(newImg, cv2.HOUGH_GRADIENT, 2, 100.0,
                                        param1=30, param2=15, minRadius=111, maxRadius=112)
        #calling drawing circle method
        #and appeding to new image list
        images.append(drawCircle(copyImage, circles))

    return images

def getEyeLid(img):
    images = []
    for i in img:
        #creates copy of original image
        copyImage = copy.deepcopy(i)
        
        #creates morphological mask 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        #dilate the image
        newImg = cv2.dilate(cv2.erode(i, kernel, iterations=1), kernel, iterations=1)
        
        #apply bluring
        newImg =  cv2.GaussianBlur(newImg, (7, 7), 1)
        #new creates morphological mask 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #dilate the image
        newImg = cv2.dilate(newImg, kernel, iterations=1)
        
        #apply bluring again
        newImg =  cv2.GaussianBlur(newImg, (3, 3), 1)

        #apply more bluring again
        newImg = cv2.medianBlur(newImg, 3)
        
        #new creates morphological mask 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #dilate the image
        newImg = cv2.dilate(newImg, kernel, iterations=1)
        
        #detect edges
        newImg = cv2.Canny(newImg, 20, 70, 11)
        
        #creating circles using hough transformation method
        circles = cv2.HoughCircles(newImg, cv2.HOUGH_GRADIENT, 2, 200.0,
                                      param1=15, param2=30, minRadius=278, maxRadius=298)
        
        #calling drawing circle method
        #and appeding to new image list
        images.append(drawCircle(copyImage, circles))
    
    return images

def main():
    #Reading all images, and creates array of readed images as BGR
    images = [cv2.imread(file) for file in glob.glob("Dataset/*.bmp")]
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in images]
    
    #getting only pupil
    pupil = getPupil(gray)
    displayImages(pupil)

    #getting only iris
    iris = getIris(gray)
    displayImages(iris)

    #getting only eyelid
    eyeLid = getEyeLid(gray)
    displayImages(eyeLid)

    #all three combined single image
    combined = [i + j + k - l - l for i, j, k, l in zip(pupil, iris, eyeLid, gray)]
    displayImages(combined)

    #saving images
    saveAsImage("pupil", pupil)
    saveAsImage("iris", iris)
    saveAsImage("eyelid", eyeLid)
    saveAsImage("combined", combined)

    
if __name__ == "__main__":
    main()

'''
    Comment Section

    In this assignment, while I researching, I learned a lot.
    I saw different techniques. For example, using machine learning
    and image processing, they were detecting points around the face
    and eyes. They were clearly getting X and Y point around eyelid.
    In addtion, they can draw line when they connect the point

    For my approach, I mainly used thresholding, morphological
    transformation and edge detecting. Using these methods, I was 
    able to get eyelid 9/10 (one case was problematic, I could reduce
    the line). I was successfully get iris and pupil 10/10. 
'''