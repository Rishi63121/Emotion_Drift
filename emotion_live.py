import cv2
import time
from deepface import DeepFace
from collections import Counter
import os
import matplotlib.pyplot as plt

# hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  #uses Haarcascades which is a pre-trained face detector to detect face at each frame.
)

cap = cv2.VideoCapture(0) # starts the webcam, 0 represents default webcam 

emotion_timeline = [] # the following are variables for tracking to finally calculate dominant emotion and stability
recent_emotions = []
frame_count = 0

start_time = time.time() # this peice of code starts the time for recording and run the cap for 30 sec.
record_duration = 30

print("Recording started...")

while True:     # while loop runs until the recording stops

    ret, frame = cap.read() # here a frame from the image is taken , ret returns True/ False saying if frame was taken or not and frame holds the dimention of image
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # now each frame is converted to grayscale images as the frames are in rgb and face detection is done better in grayscale.

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # here face is detected in a frame which uses detectMultiScale which has 3 factors (color , scaling factor , minimum neighbors)

    for (x, y, w, h) in faces: # from faces we get the corrdinates of the faces i terms if (x, y, w, h) and now they are processed

        # draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        face = frame[y:y+h, x:x+w] # the faces is cropped here for accuracy 

        # resize face for model
        face = cv2.resize(face, (224,224))

        if frame_count % 20 == 0: # here every 20 frames emotion is analysed as its a heavy task if done on every frame

            try:

                result = DeepFace.analyze( # the face prediction is done using deepface
                    face,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotion = result[0]['dominant_emotion'] # Dominant emotion is detected 

                # store last few emotions (for smoothing), last 5 prediction
                recent_emotions.append(emotion)

                if len(recent_emotions) > 5:
                    recent_emotions.pop(0)

                # get most common recent emotion
                stable_emotion = Counter(recent_emotions).most_common(1)[0][0]

                emotion_timeline.append(stable_emotion) # stores the emotion timeline 

                print("Detected:", stable_emotion)

                # show emotion on screen
                cv2.putText(frame,
                            stable_emotion,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2) # this is used to give the text on camera screen 

            except:
                pass

    cv2.imshow("Emotion Drift Analyzer", frame) # show the webcam 

    # stop recording after set time
    if time.time() - start_time > record_duration:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release() 
cv2.destroyAllWindows()

print("\nRecording finished.")

# ------------------ ANALYSIS ------------------

if len(emotion_timeline) > 0:

    emotion_counts = Counter(emotion_timeline)
    dominant_emotion = emotion_counts.most_common(1)[0]

    switches = 0

    for i in range(1, len(emotion_timeline)):
        if emotion_timeline[i] != emotion_timeline[i-1]:
            switches += 1

    stability_score = 100 - (switches / len(emotion_timeline)) * 100

    print("\n------ RESULTS ------")
    print("Dominant Emotion:", dominant_emotion[0])
    print("Emotion Switches:", switches)
    print("Stability Score:", round(stability_score,2), "%")

else:

    print("No emotions detected.")

#-------------- Graph Analysis -----------------
emotion_map = {
    'neutral':0,
    'happy':1,
    'sad':2,
    'angry':3,
    'fear':4,
    'surprise':5,
    'disgust':6
}
numeric_emotions = [emotion_map[e] for e in emotion_timeline]
plt.figure(figsize=(10,4))

plt.plot(numeric_emotions, marker='o')

plt.title("Emotion Drift Timeline")
plt.xlabel("Time Step")
plt.ylabel("Emotion")

plt.yticks(
    list(emotion_map.values()),
    list(emotion_map.keys())
)

plt.grid()

plt.show()