import face_recognition
import cv2
import imutils
import pickle
import time
import os

# Load the known faces and embeddings
data = pickle.loads(open("encodings.pickle", "rb").read())

# Initialize video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    # Grab a frame from the video stream
    ret, frame = vs.read()

    # Resize frame to process faster
    frame = imutils.resize(frame, width=500)

    # Convert the image from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the coordinates of the faces
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over each face found in the frame
    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Frame", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
vs.release()
cv2.destroyAllWindows()
