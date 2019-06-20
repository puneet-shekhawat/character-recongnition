#Importing important libraries
import numpy as np
import cv2
import operator

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True


allContoursWithData = []                # declare empty lists,
validContoursWithData = []              

npaClassifications = np.loadtxt("classifications.txt", np.float32) # read in training classifications

npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images


npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

imgTestingNumbers = cv2.imread("test1.png")          # read in testing numbers image

imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)

imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for npaContour in npaContours:                             # for each contour
    contourWithData = ContourWithData()                                             # instantiate a contour with data object
    contourWithData.npaContour = npaContour                                         # assign contour to contour with data
    contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
    contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
    contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
    allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    

for contourWithData in allContoursWithData:
    if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)

validContoursWithData.sort(key = operator.attrgetter("intRectX"))

strFinalString = ""        

for contourWithData in validContoursWithData:            # for each contour
    cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),   
              (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),    
               (0, 255, 0),2)           

    imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))            

    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))     
    npaROIResized = np.float32(npaROIResized) 

    retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1) 

    strCurrentChar = str(chr(int(npaResults[0][0])))                                 
    strFinalString = strFinalString + strCurrentChar

print ("\n" + strFinalString + "\n" )

cv2.imshow("imgTestingNumbers", imgTestingNumbers)     
cv2.waitKey(0)

cv2.destroyAllWindows()

