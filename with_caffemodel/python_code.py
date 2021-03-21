import cv2
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import serial
import time


protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
'''ArduinoSerial = serial.Serial('com4', 9600)
time.sleep(2)
print(ArduinoSerial.readline())
ArduinoSerial.write(1) #by default light is on(make it 0 for off)
'''

def main():
    cap = cv2.VideoCapture('people.mp4') # make 'people.mp4' to 0 to use your local camera
    total_frames = 0
    person = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        person = len(objects)
        while person is 0:
            '''ArduinoSerial.write(0)'''
            print("LED off")
            break
        else:
            '''ArduinoSerial.write(1)'''
            print("LED on")

        person_text = "Person: {:.2f}".format(person)
        cv2.putText(frame, person_text, (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("Output", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
