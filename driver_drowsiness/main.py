from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from camera2 import VideoCamera2
from camera3 import VideoCamera3
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import argparse
import pyttsx3


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="distracted_driver"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()

    ff=open("camst.txt","w")
    ff.write("1")
    ff.close()

    ff=open("note.txt","w")
    ff.write("0")
    ff.close()

    mycursor = mydb.cursor()
    
    if request.method=='POST':
        carno=request.form['carno']
        mycursor.execute("SELECT count(*) FROM register where carno=%s",(carno, ))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            ff1=open("cno.txt","w")
            ff1.write(carno)
            ff1.close()
            return redirect(url_for('verify_face'))
        else:
            msg="Wrong Car No.!"
        

        
    return render_template('index.html',msg=msg,act=act)



#########################
@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        
        uname=request.form['uname']
        password=request.form['pass']

        carno=request.form['carno']
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM register where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO register(id, name, mobile, email, address, uname, pass, carno, rdate, owner, utype) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)"
            val = (maxid, name, mobile, email, address, uname, password, carno, rdate, uname, 'owner')
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            #if face_st=="1":
            #    return redirect(url_for('add_photo',vid=maxid))
            #if mycursor.rowcount==1:
            #    result="Registered Success"
            #else:
            return redirect(url_for('add_photo1',vid=str(maxid)))
        else:
            result="Already Exist!"
    return render_template('register.html',result=result)

@app.route('/login1', methods=['POST','GET'])
def login1():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            msg=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            msg="Your logged in fail!!!"
                
    
    return render_template('login1.html',msg=msg)


@app.route('/login_owner', methods=['POST','GET'])
def login_owner():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where uname=%s && pass=%s && utype='owner'",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            ff1=open("log.txt","w")
            ff1.write(username1)
            ff1.close()
            msg=" Your Logged in sucessfully**"
            return redirect(url_for('owner_home')) 
        else:
            msg="Your logged in fail!!!"
                
    
    return render_template('login_owner.html',msg=msg)

@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    mycursor = mydb.cursor()

    
    return render_template('admin.html')

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    
    ff=open("user.txt","w")
    ff.write("1")
    ff.close()

    ff=open("camst.txt","w")
    ff.write("2")
    ff.close()

    vid = request.args.get('vid')
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    elif vid=="4":
        drname="normal"
        
    if request.method=='GET':
        
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()

    cursor = mydb.cursor()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+str(vid)+".jpg"
        

        cursor.execute('delete from vt_dd WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=str(vid)+"_"+str(v1)+".jpg"
        ik=2
        while ik<vv:
            cursor.execute("SELECT max(id)+1 FROM vt_dd")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            timg=str(vid)+"_"+str(ik)+".jpg"
            print(timg)
            #########
            
            # construct the argument parse 
            '''parser = argparse.ArgumentParser(
                description='Script to run MobileNet-SSD object detection network ')
            parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
            parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                              help='Path to text network file: '
                                                   'MobileNetSSD_deploy.prototxt for Caffe model or '
                                                   )
            parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                             help='Path to weights: '
                                                  'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                                  )
            parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
            args = parser.parse_args()

            # Labels of Network.
            classNames = { 0: 'background',
                1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
                5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
                10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'plastic', 18: 'sofa', 19: 'cellphone', 20: 'tvmonitor' }

            # Open video file or capture device. 
            #if args.video:
            #    cap = cv2.VideoCapture(args.video)
            #else:
            #    cap = cv2.VideoCapture(0)

            #Load the Caffe model 
            net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

            #while True:
            # Capture frame-by-frame
            #ret, frame = cap.read()
            frame = cv2.imread("static/"+drname+"/"+timg)
            frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

            # MobileNet requires fixed dimensions for input image(s)
            # so we have to ensure that it is resized to 300x300 pixels.
            # set a scale factor to image because network the objects has differents size. 
            # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
            # after executing this command our "blob" now has the shape:
            # (1, 3, 300, 300)
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            #Set to network the input blob 
            net.setInput(blob)
            #Prediction of network
            detections = net.forward()

            #Size of frame resize (300x300)
            cols = frame_resized.shape[1] 
            rows = frame_resized.shape[0]

            #For get the class and location of object detected, 
            # There is a fix index for class, location and confidence
            # value in @detections array .
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2] #Confidence of prediction 
                if confidence > args.thr: # Filter prediction 
                    class_id = int(detections[0, 0, i, 1]) # Class label

                    # Object location 
                    xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)
                    
                    # Factor for scale to original size of frame
                    heightFactor = frame.shape[0]/300.0  
                    widthFactor = frame.shape[1]/300.0 
                    # Scale object detection to frame
                    xLeftBottom = int(widthFactor * xLeftBottom) 
                    yLeftBottom = int(heightFactor * yLeftBottom)
                    xRightTop   = int(widthFactor * xRightTop)
                    yRightTop   = int(heightFactor * yRightTop)
                    # Draw location of object  
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                  (0, 255, 0))
                    try:
                                
                        image = cv2.imread("static/"+drname+"/"+timg)
                        cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                        
                        cv2.imwrite("static/"+drname+"/a"+timg, cropped)
                        #mm2 = PIL.Image.open('static/'+drname+'/'+timg)
                        #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                        #rz.save('static/'+drname+'/'+timg)
                    except:
                        print("none")
                        #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                    # Draw label and confidence of prediction in frame resized
                    if class_id in classNames:
                        label = classNames[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                             (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                             (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        #print(label) #print class and confidence'''

            
            #########
            
            sql = "INSERT INTO vt_dd(id, vid, vimage) VALUES (%s, %s, %s)"
            val = (maxid, vid, timg)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            ik+=1

        
            
        #cursor.execute('update register set fimg=%s WHERE id = %s', (vface1, vid))
        #mydb.commit()
        #shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    #cursor = mydb.cursor()
    #cursor.execute("SELECT * FROM register")
    #data = cursor.fetchall()
    return render_template('add_photo.html',vid=vid)


@app.route('/add_photo1',methods=['POST','GET'])
def add_photo1():
    vid = request.args.get('vid')
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM register where id=%s",(vid,))
    dd = cursor.fetchone()
    owner=dd[5]
    print(owner)
    if request.method=='GET':
        
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface,owner) VALUES (%s, %s, %s,%s)"
            val = (maxid, vid, vface,owner)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update register set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('login_owner',vid=vid,act='success'))
        
    
    return render_template('add_photo1.html',vid=vid)

@app.route('/add_photo2',methods=['POST','GET'])
def add_photo2():
    vid = request.args.get('vid')
    act = request.args.get('act')
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()


    print(vid)

    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM register where id=%s",(vid,))
    dd = cursor.fetchone()
    owner=dd[9]
    print(dd[9])
    
    if request.method=='GET':
        
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface,owner) VALUES (%s, %s, %s,%s)"
            val = (maxid, vid, vface, owner)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_driver'))
        
    
    return render_template('add_photo2.html',vid=vid,act=act)


###Preprocessing
@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid = request.args.get('vid')

    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"

    
    value=[]
    if request.method=='GET':
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            #Read image
            path="static/"+drname+"/"+rs[2]
            path2="static/"+drname+"/g"+rs[2]
            
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            ##RGB to Grey Scale conversion
            img = cv2.imread(path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            ##Resize image
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
            rz.save(path2)
            # Remove noise (Denoise)
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/"+drname+"/b"+rs[2]
            segment.save(path3)
            
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/"+drname+"/c"+rs[2]
            edged.save(path4)
            ##
            shutil.copy('static/assets/img/hero/11.png', 'static/'+drname+'/d'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid,drname=drname)


##Region Proposal Network (RPN)
def rpn_segmenter(model_def_file, model_file, gpu_device, inputs):
    
    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."
    
    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)

##Local Binary Pattern --Feature Extraction
def LBP(input_file, output_file, gpu_device):
    """ Runs the FR-CNN 
    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256)
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    output_im.save(output_file)

##FRCNN DD Classification
def FR_CNN(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted and classified')
        else:
                print('none')
                
@app.route('/view_photo1',methods=['POST','GET'])
def view_photo1():
    vid = request.args.get('vid')
    value=[]
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo1.html', result=value,vid=vid,drname=drname)

@app.route('/view_photo2',methods=['POST','GET'])
def view_photo2():
    vid = request.args.get('vid')
    value=[]
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"
    if request.method=='GET':
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo2.html', result=value,vid=vid,drname=drname)    

@app.route('/view_photo3',methods=['POST','GET'])
def view_photo3():
    vid = request.args.get('vid')
    value=[]
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo3.html', result=value,vid=vid,drname=drname)

@app.route('/view_photo4',methods=['POST','GET'])
def view_photo4():
    vid = request.args.get('vid')
    value=[]
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_dd where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo4.html', result=value,vid=vid,drname=drname)

@app.route('/message',methods=['POST','GET'])
def message():
    vid = request.args.get('vid')
    name=""
    drname="normal"
    if vid=="1":
        drname="calling"
    elif vid=="2":
        drname="texting"
    elif vid=="3":
        drname="behind"
    else:
        drname="normal"
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        #mycursor.execute("SELECT name FROM register where id=%s",(vid, ))
        #name = mycursor.fetchone()[0]
    return render_template('message.html',vid=vid)


@app.route('/owner_home',methods=['POST','GET'])
def owner_home():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    rr = mycursor.fetchone()
   
    
    return render_template('owner_home.html',rr=rr)

@app.route('/update',methods=['POST','GET'])
def update():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    rr = mycursor.fetchone()
    if request.method=='POST':
        mobile=request.form['mobile']
        email=request.form['email']
        ycursor.execute('update register set mobile=%s,email=%s WHERE uname = %s', (uname, ))
        mydb.commit()
        return redirect(url_for('owner_home'))
    return render_template('update.html',rr=rr)

@app.route('/view_alert',methods=['POST','GET'])
def view_alert():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    rr = mycursor.fetchone()

    mycursor.execute("SELECT * FROM vt_alert where uname=%s order by id desc",(uname,))
    data = mycursor.fetchall()
   
    
    return render_template('view_alert.html',rr=rr,data=data)

@app.route('/view_driver',methods=['POST','GET'])
def view_driver():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    rr = mycursor.fetchone()

    mycursor.execute("SELECT * FROM register where owner=%s",(uname,))
    data = mycursor.fetchall()
   
    
    return render_template('view_driver.html',rr=rr,data=data)

@app.route('/reg_driver',methods=['POST','GET'])
def reg_driver():
    ff1=open("log.txt","r")
    uname=ff1.read()
    ff1.close()
    #if 'username' in session:
    #    uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    rr = mycursor.fetchone()
    carno=rr[7]

    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        
        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO register(id, name, mobile, email, address, uname, pass, carno, rdate,owner,utype) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)"
        val = (maxid, name, mobile, email, address, name, '', carno, rdate,uname,'user')
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()            
        print(mycursor.rowcount, "record inserted.")
        #if face_st=="1":
        #    return redirect(url_for('add_photo',vid=maxid))
        #if mycursor.rowcount==1:
        #    result="Registered Success"
        #else:
        return redirect(url_for('add_photo2',vid=str(maxid)))
        
    return render_template('reg_driver.html',rr=rr)

@app.route('/verify_face',methods=['POST','GET'])
def verify_face():
    vid=""
    act=""
    ff=open("cno.txt","r")
    cno=ff.read()
    ff.close()

    ff=open("camst.txt","w")
    ff.write("1")
    ff.close()

    ff=open("check.txt","w")
    ff.write("1")
    ff.close()

    ff=open("check1.txt","w")
    ff.write("1")
    ff.close()
    
    st=""
    result=[]
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM register where carno=%s && utype='owner'",(cno,))
    dd = cursor.fetchone()
    owner=dd[5]
     
    cursor.execute("SELECT * FROM vt_face where owner=%s",(owner,))
    data = cursor.fetchall()

    
    if request.method=='POST':
        try:
            cutoff=12
            for ds in data:
                print(ds[2])
                hash0 = imagehash.average_hash(Image.open("static/frame/"+ds[2])) 
                hash1 = imagehash.average_hash(Image.open("faces/f1.jpg"))
                cc1=hash0 - hash1
                print("ccf="+str(cc1))
                if cc1<=cutoff:
                    st="yes"
                    vid=ds[1]
                    ff1=open("driver.txt","w")
                    ff1.write(str(vid))
                    ff1.close()
                    break
                else:
                    st="no"

            if st=="yes":
                act="yes"
                cursor.execute("SELECT * FROM register where id=%s",(vid,))
                result = cursor.fetchone()
        
            else:
                act="no"

        except:
            print("try")

       
    return render_template('verify_face.html',vid=vid,act=act,result=result)

@app.route('/verify',methods=['POST','GET'])
def verify():
    vid=""
    act=""
    ff=open("driver.txt","r")
    vid=ff.read()
    ff.close()
    st=""
    result=[]
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM register where id=%s",(vid,))
    dd = cursor.fetchone()
    owner=dd[9]
    driver=dd[1]

    return render_template('verify.html',vid=vid,driver=driver)

def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()
    
@app.route('/dd_verify',methods=['POST','GET'])
def dd_verify():
    vid=""
    act=""
    ff=open("driver.txt","r")
    vid=ff.read()
    ff.close()
    st=""
    stt=""
    smt=""
    mess=""
    result=[]
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM register where id=%s",(vid,))
    dd = cursor.fetchone()
    owner=dd[9]
    driver=dd[1]
     
    cursor.execute("SELECT * FROM register where uname=%s && utype='owner'",(owner,))
    row = cursor.fetchone()
    name=row[1]
    mobile=row[2]
    
    ff=open("check.txt","r")
    cc=ff.read()
    ff.close()

    ff=open("check1.txt","r")
    cc1=ff.read()
    ff.close()

    if cc=="1" and cc1=="2":
        st="1"
    else:
        if cc=="2":            
            mess="Face Not Detected"
        elif cc=="3" and cc1=="3":
            act="1"
            stt="Alert: Driver Drowsiness, Eye Closed and Yawning"
            mess="Eye Closed, yawning"
            speak(stt)
        elif cc1=="3":
            act="1"
            stt="Alert: Driver Drowsiness, Yawning"
            mess="Driver yawning"
            speak(stt)
        elif cc=="3":
            act="1"
            stt="Alert: Driver Drowsiness, Eye Closed"
            mess="Eye Closed"
            speak(stt)
        #if act=="1":
        #        speak(stt)
        
        
        st="2"

    fm="myface.png"
    drname="calling"
    if request.method=='POST':
        s=1
        '''for rs in rr:

            fn="a"+rs[2]
            vid=rs[1]
            if vid=="1":
                drname="calling"
            elif vid=="2":
                drname="texting"
            elif vid=="3":
                drname="behind"
            else:
                drname="normal"'''

    
        
        #path="static/"+drname+"/"+fn
        # construct the argument parse 
        '''parser = argparse.ArgumentParser(
            description='Script to run MobileNet-SSD object detection network ')
        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                               )
        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Labels of Network.
        classNames = { 0: 'background',
            1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
            5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'plastic', 18: 'sofa', 19: 'cellphone', 20: 'tvmonitor' }

        # Open video file or capture device. 
        

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        frame = cv2.imread("myface.png")
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                try:
                            
                    image = cv2.imread("myface.png")
                    cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                    
                    cv2.imwrite("crop.png", cropped)
                    #mm2 = PIL.Image.open("crop.png").convert('L')
                    #mm2.save("crop.png")
                    #mm2 = PIL.Image.open('static/'+drname+'/'+timg)
                    #rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                    #rz.save('static/'+drname+'/'+timg)
                except:
                    print("none")
                    #shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    #print(label) #print class and confidence'''

        try:
            #########
            cutoff=10
            cc1=0
            cc2=0
            cc3=0
            cc4=0
            cursor.execute("SELECT * FROM vt_dd")
            rr = cursor.fetchall()
            for ds in rr:
                print(ds[2])
                if ds[1]==1:
                    hash0 = imagehash.average_hash(Image.open("static/calling/"+ds[2])) 
                    hash1 = imagehash.average_hash(Image.open("faces/f1.jpg"))
                    cc1=hash0 - hash1
                    print("cc="+str(cc1))
                    if cc1<=cutoff:
                        act="1"
                        stt="Alert: Driver Calling"
                        break
                elif ds[1]==4:
                    act="2"
                    st="Normal"
                elif ds[1]==2:
                    hash20 = imagehash.average_hash(Image.open("static/texting/"+ds[2])) 
                    hash21 = imagehash.average_hash(Image.open("faces/f1.jpg"))
                    cc2=hash20 - hash21
                    print("cc2="+str(cc2))
                    if cc2<=cutoff:
                        act="1"
                        stt="Alert: Driver Texting"
                        break

                '''elif ds[1]==3:
                    hash30 = imagehash.average_hash(Image.open("static/behind/"+ds[2])) 
                    hash31 = imagehash.average_hash(Image.open("faces/f1.jpg"))
                    cc3=hash30 - hash31
                    print("cc3="+str(cc3))
                    if cc3<=cutoff:
                        act="1"
                        st="Alert: Looking Behind"
                        break'''
            

            print("act="+act)
        except:
            print("try")

            
    smt=""
    if act=="1":
        mess=stt+" "+driver
        print(mess)
        cursor.execute("SELECT max(id)+1 FROM vt_alert")
        maxid = cursor.fetchone()[0]
        if maxid is None:
            maxid=1
        sql = "INSERT INTO vt_alert(id, uname, driver, message) VALUES (%s, %s, %s, %s)"
        val = (maxid, owner, driver, stt)
        cursor.execute(sql, val)
        mydb.commit()

        ff=open("note.txt","r")
        nn=ff.read()
        ff.close()
        n=int(nn)
        if n<=3:
            m=n+1
            mm=str(m)
            ff=open("note.txt","w")
            ff.write(mm)
            ff.close()
            smt="1"
            #url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
            #webbrowser.open_new(url)
        
          
    return render_template('dd_verify.html',vid=vid,act=act,st=st,stt=stt,smt=smt,mess=mess,mobile=mobile,name=name)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))


##################
def gen(camera):
    
    while True:
        frame = camera.get_frame()
   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

###################
def gen2(camera2):
    
    while True:
        frame2 = camera2.get_frame2()
   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
    
@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

############
def gen3(camera3):
    
    while True:
        frame3 = camera3.get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n\r\n')
    
@app.route('/video_feed3')
def video_feed3():
    return Response(gen3(VideoCamera3()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(track_alertness(), mimetype='multipart/x-mixed-replace; boundary=frame')
#######################

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
