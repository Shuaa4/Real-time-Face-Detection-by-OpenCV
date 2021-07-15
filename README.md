# Real-time-Face-Detection-by-OpenCV
This project is done using Pycharm IDE and Python, it is a Real-Time Face recognition using OpenCV while performing object detection using Haar feature-based cascade classifiers for detecting face, eyes, and smile.
# 1. Import and initialize
Start by importing OpenCV and create a directory (ex: Cascades) to gather all Haar classifiers files that you want to use in you project, then use their path to load them into your project.
import cv2
 
faceCascade = "cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_FRONTALFACE_DEFAULT.xml')

# 2. Setting up your camera
To start we need to capture the face and to do so we are using the PC embedded camera which we are referring to it using (0) & setting the window size to specific measures in the following code lines:
cap = cv2.VideoCapture(0)  # write 0 to use the pc embedded camera 
cap.set(3,640) # set Width 
cap.set(4,380) # set Height

# 3. Call the classifier function
We will set our camera and inside the loop, load our input video in grayscale mode then we must call our classifier function, passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face.
while True: 
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = faceCascade.detectMultiScale( 
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(20, 20) 
    )
   # 4. Detecting Faces
The function will detect faces on the image. Next, we must "mark" the faces in the image, using, for example, a blue rectangle. If faces are found, it returns the positions of detected faces as a rectangle with the left up corner (x,y) and having "w" as its Width and "h" as its Height ==> (x,y,w,h). This is done with this portion of the code:

for face in faces: 
        top, right, bottom, left = face 
        # draw a blue rectangle around the face 
        cv2.rectangle(img, (top, right), (top + bottom, right + left), (255, 0, 0), 2) 
        roi_gray = gray[right:right + left, top:top + bottom] 
        roi_color = img[right:right + left, top:top + bottom]
        
# 5. Final Touches
If the user wants to quit the program, the button ESC is set to terminate the program in the following code lines, and to alert the user there is a small message shown on the top-right corner of the detected face boarders.
k = cv2.waitKey(30) & 0xff 
    if k == 27:  # press 'ESC' to quit 
        break 
cap.release() 
cv2.destroyAllWindows()
