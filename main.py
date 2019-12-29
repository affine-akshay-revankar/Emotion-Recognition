from flask import Flask, render_template, Response, request, redirect,url_for
import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras import backend as K
import tensorflow as tf

app = Flask(__name__)


# parameters for loading data and images
face_detection_model_path = 'haarcascade_file/haarcascade_frontalface_default.xml'
emotion_model_path = 'model/_mini_XCEPTION.102-0.66.hdf5'

# # loading models
face_detection = cv2.CascadeClassifier(face_detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)

def loadmodel():
	global emotion_classifier
	emotion_classifier = load_model(emotion_model_path, compile=False)
            # this is key : save the graph after loading the model
	# global graph
	# graph = tf.get_default_graph()

# list of models
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

def gen_frames(camera_id,cap):
        # loading models
        # face_detection = cv2.CascadeClassifier(face_detection_model_path)
        # emotion_classifier = load_model(emotion_model_path, compile=False)
    # In case you want to detect emotions on a video, provide the video file path instead of 0 for VideoCapture.
        # cap = cv2.VideoCapture(0)

        #    while True:
        ret, frame = cap.read()
        if not ret:
            print('no frame')

        print('1')

        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        print('length faces outside: ',len(faces))
        # print('faces type',type(faces))
        if len(faces)>0:
            print('2')
            print('length face:',len(faces))
            # draw box around faces
            for face in faces:
                (x,y,w,h) = face
                
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                print('3')
                # emotion_classifier = load_model('model/_mini_XCEPTION.102-0.66.hdf5', compile=False)

                # with graph.as_default():
                result = emotion_classifier.predict(roi)[0]
                
                print(result)
                frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2)

                print('4')

                if result is not None:
                    print('5')
                    emotion_index = np.argmax(result)
                    print(EMOTIONS[emotion_index])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,EMOTIONS[emotion_index],(x+5,y-35), font, 0.6,(255,0,0),2,cv2.LINE_AA) 
                    # cv2.imshow('Video', frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
                print('6')
        jpeg = cv2.imencode('.jpg', frame)[1]
            # jpeg_sa_text= base64.b64encode('.jpg', frame)
            # return jpeg_sa_text
            # jpegimg = jpeg.tobytes()
        return jpeg.tobytes()
                # frame1 = jpeg.tobytes()
        # cap.release()
        # cv2.destroyAllWindows()

def gen(camera_id):


    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
   
    while True:
        frame = gen_frames(camera_id,cap)
        if not frame:
            continue
        print('frame')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed/<camera_id>/', methods=["GET"])
def video_feed(camera_id):

    # loadmodel()
    global emotion_classifier
    emotion_classifier = load_model(emotion_model_path, compile=False)
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port='5002')
    app.run(host='0.0.0.0', port=8081)
