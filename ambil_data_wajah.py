import cv2
import os

webCam = cv2.VideoCapture(0)
currentframe = 0
name = input("Masukkan nama: ")
path = './' + name
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
    success, frame = webCam.read()
    faces = detect_bounding_box(frame)
    cv2.imshow("Output", frame)
    if len(faces) == 0:
        pass
    else:
        cv2.imwrite(os.path.join(path , name + str(currentframe) + '.jpg'), frame)
        currentframe += 1
        print('Gambar ke ' + str(currentframe) + 'berhasil ditangkap')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webCam.release()
cv2.destroyAllWindows()