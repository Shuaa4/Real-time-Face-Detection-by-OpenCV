import cv2 
 
faceCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_FRONTALFACE_DEFAULT.xml') 
 
cap = cv2.VideoCapture(0)  # write 0 to use the pc embedded camera 
cap.set(3,640) # set Width 
cap.set(4,380) # set Height 
while True: 
    ret, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = faceCascade.detectMultiScale( 
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(20, 20) 
    ) 
    for face in faces: 
        top, right, bottom, left = face 
        # draw a blue rectangle around the face 
        cv2.rectangle(img, (top, right), (top + bottom, right + left), (255, 0, 0), 2) 
        roi_gray = gray[right:right + left, top:top + bottom] 
        roi_color = img[right:right + left, top:top + bottom] 
     
    cv2.imshow('Face Detection - Reem', img) 
    k = cv2.waitKey(30) & 0xff 
    if k == 27:  # press 'ESC' to quit 
        break 
cap.release() 
cv2.destroyAllWindows()
