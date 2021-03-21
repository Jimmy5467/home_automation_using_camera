
import cv2
import serial  # Serial imported for Serial communication
import time  # Required to use delay functions


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
#ArduinoSerial = serial.Serial('com18', 9600) #find your own com port
#time.sleep(2)
#print(ArduinoSerial.readline())
#ArduinoSerial.write('1')

cap = cv2.VideoCapture('people.mp4')


while True:
    success, img = cap.read()

    faces = faceCascade.detectMultiScale(img, 1.1, 1)
    var = 1
    for (x, y, w, h) in faces:
        var += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
        #ArduinoSerial.write(1)
    else:
        #ArduinoSerial.write(0)
        print('LED off')

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')
