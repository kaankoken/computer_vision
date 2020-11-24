import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt 

#Display method for images
def displayImages(img):
    #iterates all the images
    count: int = 1
    for i in img:
        cv2.imshow(str(count), mat=i)
        count+=1
    if cv2.waitKey(0) and 0xff == 27:
        cv2.destroyAllWindows() 

def grayWorld(img):
   
    ''' Quote from 3rd slide part_1
    The image average r average, g average, b average is gray
    Use weights 1/r average, 1/g average, 1/b average
    '''
    image = []
    for i in img:
        #split to the BGR      
        b, g, r = cv2.split(i)
        
        #mean of 3 channels
        mean_b = np.mean(np.mean(b))
        mean_g = np.mean(np.mean(g))
        mean_r = np.mean(np.mean(r))

        #weigths of channels
        x = (1 / r)
        y = (1 / b)
        z = (1 / g)
        
        #apply grayWorld solution for illumination problem
        r = (x * r + r).astype(np.uint8)
        g = (z * g + g).astype(np.uint8)
        b = (y * b + b).astype(np.uint8)
        
        #combine image
        i = cv2.merge((b,g,r))
        #add to new image list
        image.append(i)
    
    return image

def test(img):
    image = []
    for i in img:
        b, g, r = cv2.split(i)

        mean_b = np.mean(np.mean(b))
        mean_g = np.mean(np.mean(g))
        mean_r = np.mean(np.mean(r))

        alpha = mean_g / mean_r
        beta = mean_g / mean_b

        r = (alpha * r)
        b = (beta * b).astype(np.uint8)

        x = cv2.merge((b, g, r))
        image.append(x)

    return image

def threshold(occurency:list, img):
    #threshold value, get max occurred one
    sort = np.argsort(occurency)
    
    #thresholding image according to threshold value
    ret, new_img = cv2.threshold(img, sort[-1], 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return new_img

def histogram_bgr(img):
    thresholded_image = []
    for i in img:
        #calculates histogram values and display histogram
        b, g, r = cv2.split(i)
        occurency_b, bins = np.histogram(b.ravel(),255,[0,255])
        occurency_g, bins = np.histogram(g.ravel(),255,[0,255])
        occurency_r, bins = np.histogram(r.ravel(),255,[0,255])

        #plt.hist(b.ravel(),255,[0,255]);
        #plt.show()
        b, g, r = threshold(occurency_b, b), threshold(occurency_g, g), threshold(occurency_r, r)
        i = cv2.merge((b, g, r))
        thresholded_image.append(i)
    return thresholded_image

def toYCbCr(img):
    image = []
    for i in img:
        #split the channels to BGR
        b,g,r = cv2.split(i)

        #Convert BGR channels to Y Cb Cr values
        y = (.299*r + .587*g + .114*b).astype(np.uint8)
        cb = (128 -.168736*r -.331364*g + .5*b).astype(np.uint8)
        cr = (128 +.5*r - .418688*g - .081312*b).astype(np.uint8)

        #combine as YCbCr
        i = cv2.merge((y, cb, cr))
        
        image.append(i)
    return image

def toRGB(y, cb, cr):
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    
    return r, g, b

def main():
    #Reading all images, and creates array of readed images as BGR
    images = [cv2.imread(file) for file in glob.glob("Dataset/*.png")]
   
    #call grayWorld method
    enlightened_bgr = grayWorld(images)
    displayImages(enlightened_bgr)
    
    #call histogram method
    thresholded_bgr= histogram_bgr(enlightened_bgr)

    #display BGR image
    displayImages(thresholded_bgr)

    #convert BGR image to YCrCb since it is BGR not RGB
    yCbCr = toYCbCr(enlightened_bgr)

    #thresolding YCbCr values
    thresholded_ycbcr = histogram_bgr(yCbCr)
    
    #display YCbCr image
    displayImages(thresholded_ycbcr)

    new_yCbCr = []
    #convert YCbCr into BGR
    for i in thresholded_ycbcr:
        y, cb, cr = cv2.split(i)
        r,g,b = toRGB(y, cb, cr)
        i = cv2.merge((b,g,r))
        new_yCbCr.append(i)

    #display converted YCbCr image
    displayImages(new_yCbCr)

if __name__ == "__main__":
    main()

'''
     Comment Section

    I experimented both RGB and YCbCr images. Besides the background color problem of images,
    I think YCbCr gives the best result by looking at the result because Y channel is
    directly about the lumination. Therefore, it allows to manipulate image directly.

    Also the background problem that I have could be solved with manual adjustment by looking the
    local maxima values. Since I could not find a way to detect 2nd local maxima that I need
    programmatically, I did not prefer to do manual adjustment.
'''