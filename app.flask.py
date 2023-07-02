from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model_face3.pkl', 'rb'))


facenet = FaceNet()
face_embedding = np.load('2_classes_done_tambah_1.npz')

Y = face_embedding['arr_1']

unknown_threshold = -0.16

encoder = LabelEncoder()

encoder.fit(Y)

face_classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the Haar cascade classifier for face detection

# Open the camera

# Set the camera resolution

latest_name_final = "unknown"

# Set the font for displaying the class name

def generate_frames():
    global latest_name_final
    
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():  # Memeriksa apakah kamera berhasil terbuka
        print("Kamera tidak dapat diakses")
    
    while True:
        ret,frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(gray_img, 1.3, 5)
        if len(face) == 0:
            latest_name_final = "unknwon"
        else:
            for x,y,w,h in face:
                img = rgb_img[y : y + h, x : x + w]
                img = cv.resize(img, (160, 160))
                img = np.expand_dims(img, axis = 0)
                y_pred = facenet.embeddings(img)
                name = model.predict(y_pred)
                distance = model.decision_function(y_pred)
                min_distance = np.min(distance)
                if min_distance > unknown_threshold:
                    name_final = "unknown"
                else:
                    name_final = encoder.inverse_transform(name)
                name_final_pol = str(name_final)
                latest_name_final = ''.join(name_final)
                cv.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 255), 10)
                cv.putText(frame, name_final_pol, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)

        # Display the resulting frame
        # cv2.imshow('Face Detection', frame)
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Exit if the 'q' key is pressed

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

# # Release the capture and close the window
# cap.release()
# cv2.destroyAllWindows()
@app.route('/video_face')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/name_final')
def get_name_final():
    global latest_name_final
    return jsonify({'name_final': latest_name_final})

if __name__ == "__main__":
    app.run(debug=True)
