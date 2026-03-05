import cv2                          # OpenCV library for image processing and webcam capture
import time                         # Used to track recording duration
from deepface import DeepFace      # DeepFace library for emotion detection
from collections import Counter    # Helps count frequency of detected emotions
import os                           # Used for setting environment variables
import matplotlib.pyplot as plt    # Used to plot the emotion timeline graph
import streamlit as st             # Streamlit library to build the web interface

# hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  
# Sets TensorFlow log level (0 = show logs, 3 would hide warnings)

st.set_page_config(page_title="Emotion Drift Analyzer", layout="wide")
# Configures the Streamlit page title and layout (wide mode)

st.title("Emotion Drift Analyzer")
# Displays the main title on the Streamlit web page

st.write("Analyze emotional changes over time using webcam input.")
# Displays a short description under the title

record_duration = st.slider("Recording Duration (seconds)", 10, 60, 30)
# Creates a slider where the user selects recording duration
# Minimum = 10 seconds, Maximum = 60 seconds, Default = 30 seconds

# Load face detector
if st.button("Start Emotion Analysis"):
# Starts the emotion detection process when the button is clicked

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )  
    # Loads a pre-trained Haar Cascade face detection model from OpenCV

    cap = cv2.VideoCapture(0)  
    # Starts the webcam
    # "0" means the default system webcam

    emotion_timeline = []  
    # List to store emotions detected over time

    recent_emotions = []  
    # Stores the last few detected emotions to smooth predictions

    frame_count = 0  
    # Keeps track of the number of frames processed

    start_time = time.time()  
    # Records the start time of recording

    status = st.empty()  
    # Creates a placeholder in Streamlit to update status messages

    progress = st.progress(0)  
    # Creates a progress bar starting at 0%

    status.info("Recording started...")
    # Displays a message that recording has begun

    while True:     
    # Infinite loop that continuously captures webcam frames

        ret, frame = cap.read()  
        # Reads a frame from the webcam
        # ret = True if frame captured successfully
        # frame = actual image captured

        if not ret:
            break
        # If the frame is not captured, stop the loop

        frame_count += 1  
        # Increase frame counter

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        # Converts frame from color (BGR) to grayscale
        # Face detection works faster on grayscale images

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
        # Detects faces in the frame
        # 1.3 = scaling factor (image size reduction per scan)
        # 5 = minimum neighbors required to confirm a face

        for (x, y, w, h) in faces:  
        # Loops through each detected face
        # x,y = top-left corner of face
        # w,h = width and height of face

            # draw box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            # Draws a green rectangle around the detected face

            face = frame[y:y+h, x:x+w]  
            # Crops the face region from the frame

            # resize face for model
            face = cv2.resize(face, (224,224))
            # Resizes the face to 224x224 pixels
            # This size works well with DeepFace models

            if frame_count % 20 == 0:  
            # Runs emotion detection every 20 frames
            # This reduces computational load

                try:

                    result = DeepFace.analyze(
                        face,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    # Uses DeepFace to analyze the cropped face
                    # actions=['emotion'] means detect only emotion
                    # enforce_detection=False allows prediction even if face detection confidence is low

                    emotion = result[0]['dominant_emotion']  
                    # Extracts the dominant emotion predicted by the model

                    # store last few emotions (for smoothing), last 5 prediction
                    recent_emotions.append(emotion)
                    # Adds detected emotion to recent emotions list

                    if len(recent_emotions) > 5:
                        recent_emotions.pop(0)
                    # Keeps only the last 5 emotions
                    # Removes the oldest emotion if list exceeds 5

                    # get most common recent emotion
                    stable_emotion = Counter(recent_emotions).most_common(1)[0][0]
                    # Finds the most frequent emotion in the last 5 predictions
                    # This stabilizes the prediction (reduces sudden changes)

                    emotion_timeline.append(stable_emotion)  
                    # Adds the stable emotion to the emotion timeline list

                    print("Detected:", stable_emotion)
                    # Prints detected emotion in terminal

                    # show emotion on screen
                    cv2.putText(frame,
                                stable_emotion,
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,255,0),
                                2)
                    # Displays the detected emotion text above the face rectangle

                except:
                    pass
                # Prevents the program from crashing if DeepFace fails on a frame

        cv2.imshow("Emotion Drift Analyzer", frame)  
        # Displays the webcam window with face box and emotion label

        elapsed = time.time() - start_time  
        # Calculates how much time has passed since recording started

        progress.progress(min(elapsed / record_duration, 1.0))
        # Updates the progress bar in Streamlit
        # Stops at 100%

        # stop recording after set time
        if elapsed > record_duration:
            break
        # Stops recording when selected duration is reached

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Allows manual stop by pressing the 'q' key


    cap.release()  
    # Releases the webcam

    cv2.destroyAllWindows()
    # Closes all OpenCV windows

    status.success("Recording Finished!")
    # Updates Streamlit status message

    print("\nRecording finished.")
    # Prints message in terminal

    # ------------------ ANALYSIS ------------------

    if len(emotion_timeline) > 0:
    # Checks if any emotions were detected

        emotion_counts = Counter(emotion_timeline)
        # Counts how many times each emotion appeared

        dominant_emotion = emotion_counts.most_common(1)[0]
        # Finds the most frequent emotion overall

        switches = 0
        # Variable to count emotion changes

        for i in range(1, len(emotion_timeline)):
            if emotion_timeline[i] != emotion_timeline[i-1]:
                switches += 1
        # Counts how many times the emotion changed between frames

        stability_score = 100 - (switches / len(emotion_timeline)) * 100
        # Calculates emotional stability
        # More switches = lower stability score

        st.subheader("Results")
        # Displays results section

        col1, col2, col3 = st.columns(3)
        # Creates three columns to display metrics

        col1.metric("Dominant Emotion", dominant_emotion[0])
        # Shows most frequent emotion

        col2.metric("Emotion Switches", switches)
        # Shows number of emotion changes

        col3.metric("Stability Score", f"{round(stability_score,2)} %")
        # Shows emotional stability percentage

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
        # Maps emotions to numeric values for graph plotting

        numeric_emotions = [emotion_map[e] for e in emotion_timeline]
        # Converts emotion labels to numbers

        fig, ax = plt.subplots(figsize=(10,4))
        # Creates a matplotlib graph figure

        ax.plot(numeric_emotions, marker='o')
        # Plots emotion values over time

        ax.set_title("Emotion Drift Timeline")
        # Graph title

        ax.set_xlabel("Time Step")
        # X-axis label

        ax.set_ylabel("Emotion")
        # Y-axis label

        ax.set_yticks(list(emotion_map.values()))
        # Sets y-axis positions

        ax.set_yticklabels(list(emotion_map.keys()))
        # Sets y-axis emotion labels

        ax.grid()
        # Adds grid lines for readability

        st.pyplot(fig)
        # Displays the graph in the Streamlit app

    else:
        st.write("No emotions detected.")
        # Displays message if no faces/emotions were detected