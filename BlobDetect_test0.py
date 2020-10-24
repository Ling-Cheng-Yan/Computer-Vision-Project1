import cv2
import numpy as np;

# Read image
test_photo = r"DJI_0001_R.JPG"
im = cv2.imread(test_photo, cv2.IMREAD_GRAYSCALE)

# Set up the detector with parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 0.1

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.65

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Filter by Color
params.filterByColor = True 
params.blobColor = 255

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)



# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)