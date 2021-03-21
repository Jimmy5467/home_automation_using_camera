import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

cap = cv2.VideoCapture('people.mp4')


while True:
    success, img = cap.read()

    faces = faceCascade.detectMultiScale(img, 1.1, 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')