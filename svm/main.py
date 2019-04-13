import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from joblib import dump, load
import os
import matplotlib.pyplot as plt

##emotions = ["anger","disgust", "fear","happy", "sad", "surprise","neutral"] #Emotion list
##emotions = [  "negative","positive","neutral" ] #Emotion list
emotions = [  "anger","fear","disgust","sad"] #Emotion list
##emotions = ["happy","surprise"] #Emotion list


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3,verbose=True)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
##    print(training)
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def distance(x1 , y1 , x2 , y2): 
  
    # Calculating distance 
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0)



def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
##        print(())
        d1=distance(shape.part(48).x, shape.part(48).y,shape.part(54).x,shape.part(54).y)
        d2=distance(shape.part(51).x, shape.part(51).y,shape.part(57).x,shape.part(57).y)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
##            print(x,y,w,z)
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
##            print(landmarks_vectorised)
##        landmarks_vectorised.append(d1)
##        landmarks_vectorised.append(d2)
##        print(len(landmarks_vectorised))    
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
##                print(training_data)
                training_labels.append(emotions.index(emotion))
        for item in prediction:
##            print(item)
            image = cv2.imread(item)
##            print(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
##                print(item,emotions.index(emotion))
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
accur_lin = []


for i in range(1,2):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
##    np.save("npar_train"+str(i),npar_train)
    npar_trainlabs = np.array(training_labels)
##    np.save("npar_trainlabs"+str(i),npar_trainlabs )
    print("training SVM poly %s" %i) #train SVM

##    print((training_data[:]),(training_labels[:]))
##    plt.scatter(training_data[:], training_labels[:], s=50, cmap='spring'); 
##    plt.show()

    npar_train= preprocessing.scale(npar_train)
    print(type(npar_train[0]))
    print(npar_train[0])
    print(npar_train)
    np.save("npar.npy",npar_train)
    clf.fit(npar_train, training_labels)
    
    dump(clf, 'svm.joblib')
    
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
##    np.save("npar_pred"+str(i),npar_pred)
    npar_pred=preprocessing.scale(npar_pred)
##    print(prediction_labels)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("poly ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
print("Mean value of svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs






'''
----------------------------------------------------------------------------------
'''


def face1(image):
    frame =image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image
##    print("d",detections)
    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
    ##        print(d)

        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)

        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_test = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
##            print(x,y,w,z)
            landmarks_test.append(w)
            landmarks_test.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_test.append(dist)
            landmarks_test.append((math.atan2(y, x)*360)/(2*math.pi))
        d1=distance(shape.part(48).x, shape.part(48).y,shape.part(54).x,shape.part(54).y)
        d2=distance(shape.part(51).x, shape.part(51).y,shape.part(57).x,shape.part(57).y)
##        landmarks_test.append(d1)
##        landmarks_test.append(d2)
##            
##        print(int(xmean),ymean)
        

            
        try:
            d1=distance(shape.part(48).x, shape.part(48).y,shape.part(54).x,shape.part(54).y)
            d2=distance(shape.part(51).x, shape.part(51).y,shape.part(57).x,shape.part(57).y)
##            print("distance",d1,d2)
        except Exception :
            print("error")
            
    cv2.imshow("image", frame) #Display the frame
    cv2.imwrite("image.jpg",frame)

    return landmarks_test

clf = load('svm.joblib')
print(clf)
print('w = ',clf.coef_)
print('b = ',clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))


##im=cv2.imread("111_3.png")
##X=face1(im)
##X=np.array([X])

##pred_pro = clf.predict_proba(X)
##print (pred_pro)
##pred_pro1 = clf.predict(X)
##print (pred_pro1)
##print(clf.score(X,X))
##clf.decision_function(X)
##clf.predict(X)


fidget_folders = [folder for folder in os.listdir('.') if 'CKtrain' in folder]
'''

print(fidget_folders)
n = 0
y_pred=[]
y_true=[]
for folder in fidget_folders:
    for imfile in os.scandir(folder):
####        print(imfile.path)
        z=imfile.path
        t=(imfile.path.split("_")[-1].split(".")[0])
        im= cv2.imread(z)
        try:
            
            X=face1(im)
        except Exception:
            continue
        X=np.array([X])
##        print(X[0])
##        print(preprocessing.scale(X[0]))
##        pred_pro = clf.predict(X)
##        print (pred_pro)
        X= preprocessing.scale(X[0])
        X=[X]
        pred_pro = clf.predict(X)
        y_pred.append(pred_pro[0])
        y_true.append(int(t))
##        print(X)
##        print (imfile.path,pred_pro[0])
        n=n+1
##        if(n==3):
##            break
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
            break
print((y_true),((y_pred)))
s=accuracy_score(y_true, y_pred)
print(s)

'''





import cv2
import dlib
#Set up some required objects

video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
c=0
while True:
    ret, frame = video_capture.read()
    im= (frame)
##    print(frame.shape[0])
    try:
        X=face1(im)
    except Exception:
        continue
    X=np.array([X])
##        print(X[0])
##        print(preprocessing.scale(X[0]))
##        pred_pro = clf.predict(X)
##        print (pred_pro)
    X= preprocessing.scale(X[0])
    X=[X]
    pred_pro = clf.predict(X)
    pred=(pred_pro[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(pred),(frame.shape[0]//2,frame.shape[1]//2), font, 4,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("image", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

