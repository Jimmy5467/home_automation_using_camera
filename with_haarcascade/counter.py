import cv2

pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Function to perform pedestrian detection from images. Pass an image as a variable.
def pedestrianDetection(frame):

    pedestrians = pedestrian_cascade.detectMultiScale( frame, 1.1, 1)
    # To draw a rectangle on each pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
        # Display frames in a window'
    return frame

cap = cv2.VideoCapture('people.mp4')
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
print("Processing Video...")
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    out.release()
    break
  output = pedestrianDetection(frame)
  out.write(output)
out.release()
print("Done processing video")