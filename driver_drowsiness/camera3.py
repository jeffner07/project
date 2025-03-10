# camera.py

import cv2
import numpy as np 
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import PIL.Image
from PIL import Image
import pyttsx3
class VideoCamera3(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        self.video = cv2.VideoCapture(0)
        self.k=1
        self.area = 0
        
    def alarm(self):
        s=1
    def find_area(self,array):
        a = 0
        ox,oy = array[0]
        for x,y in array[1:]:
            a += abs(x*oy-y*ox)
            ox,oy = x,y
        return a/2

        
    
    def __del__(self):
        self.video.release()
        
   
    def get_frame(self):
        success, image = self.video.read()
        #self.out.write(image)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
        face_cascPath = 'haarcascade_frontalface_alt.xml'  #face detect model
        faceCascade = cv2.CascadeClassifier(face_cascPath)
        eyeCascade = cv2.CascadeClassifier(eye_cascPath)
        # Read the frame
        #_, img = cap.read()
        detector = dlib.get_frontal_face_detector() 
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

        ##########
        
        #####################
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        # print("Found {0} faces!".format(len(faces)))
        ey=""
        
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_tmp = image[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
            frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            lene=len(eyes)
            print("len="+str(lene))
            if len(eyes) == 0:
                ey="close"
                
                print('eye closed!!!')
            else:
                ey="open"
                
                #print('eyes!!!')
            frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
            #cv2.imshow('Face Recognition', frame_tmp)
        ###############
        yw=""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        faces = detector(gray) 
        for face in faces: 
        # The face landmarks code begins from here 
            x1 = face.left() 
            y1 = face.top() 
            x2 = face.right() 
            y2 = face.bottom() 
            # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3) 
            landmarks = predictor(gray, face) 
            
            # We are then accesing the landmark points 
            Landarkss = [] 
            for n in range(48, 55): 
                x = landmarks.part(n).x 
                y = landmarks.part(n).y 
                #print("Landmark1--->",(x,y))
                Landarkss.append((x,y))
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1) 
            Landarkss2 = [] 
            for n in range(54, 61): 
                x = landmarks.part(n).x 
                y = landmarks.part(n).y 
                #print("Landmark2--->",(x,y))
                Landarkss.append((x,y))
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1) 
            #print("Landmarks--->",Landarkss)

            array = Landarkss + Landarkss2
            #print("Landmarks--->",array)
            self.area = self.find_area(array)
            #print("Total area--->", area)
            print(self.area)
            YAWN_THRESH = 29000
            if (self.area > YAWN_THRESH):
                #alarm()
                yw="yes"
                print("You yawned")  
            else:
                yw="no"
                print("yw no")
                #pass
        ##############
        # Draw the rectangle around each face
        j = 1
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
         
        for (x, y, w, h) in faces:
            mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("static/myface.jpg", mm)
            j += 1

        if ey=="close":
            ff=open("check.txt","w")
            ff.write("3")
            ff.close()
        elif j<=1:
            ff=open("check.txt","w")
            ff.write("2")
            ff.close()
        
        else:
            ff=open("check.txt","w")
            ff.write("1")
            ff.close()

        ##
        print(self.k)
        if yw=="yes":
            ff=open("check1.txt","w")
            ff.write("3")
            ff.close()
            self.k=1
        else:
            self.k+=1
            if self.k>100:
                ff=open("check1.txt","w")
                ff.write("2")
                ff.close()
            
            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
