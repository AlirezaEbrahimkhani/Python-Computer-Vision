import cv2 as cv
import matplotlib.pyplot as plt

def main():
    image = cv.imread("../assets/grey-image.jpg")
    image = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
    
    dst = cv.equalizeHist(image)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(dst)
    plt.show()
    
main()