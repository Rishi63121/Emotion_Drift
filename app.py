import streamlit as st
import cv2
import time
from deepface import DeepFace
from collections import Counter
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Emotion Drift Analyzer", layout="wide")

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align:center;'>🧠 Emotion Drift Analyzer</h1>
<p style='text-align:center;color:gray;'>Real-time emotion tracking using webcam</p>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")
duration = st.sidebar.slider("Recording Duration (seconds)", 10, 60, 20)

start = st.sidebar.button("▶ Start Analysis")

# ------------------ MAIN PLACEHOLDERS ------------------
col1, col2 = st.columns([2,1])

video_placeholder = col1.empty()
graph_placeholder = col2.empty()

status = st.empty()

# ------------------ MAIN LOGIC ------------------
if start:

    cap = cv2.VideoCapture(0)

    emotion_timeline = []
    recent_emotions = []

    emotion_map = {
        'neutral':0, 'happy':1, 'sad':2,
        'angry':3, 'fear':4, 'surprise':5, 'disgust':6
    }

    start_time = time.time()

    status.info("🎥 Recording started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (500, 350))

        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']

            # smoothing
            recent_emotions.append(emotion)
            if len(recent_emotions) > 5:
                recent_emotions.pop(0)

            stable_emotion = Counter(recent_emotions).most_common(1)[0][0]
            emotion_timeline.append(stable_emotion)

            # overlay text
            cv2.rectangle(frame, (0,0), (500,50), (0,0,0), -1)
            cv2.putText(frame, f"Emotion: {stable_emotion}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        except:
            pass

        # -------- DISPLAY VIDEO --------
        video_placeholder.image(frame, channels="BGR")

        # -------- LIVE GRAPH --------
        if len(emotion_timeline) > 1:
            numeric = [emotion_map[e] for e in emotion_timeline]

            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(numeric, marker='o')

            ax.set_yticks(list(emotion_map.values()))
            ax.set_yticklabels(list(emotion_map.keys()))

            ax.set_title("Live Emotion Drift")
            ax.grid()

            graph_placeholder.pyplot(fig)
            plt.close(fig)

        # stop condition
        if time.time() - start_time > duration:
            break

    cap.release()

    # clear camera
    video_placeholder.empty()

    status.success("✅ Recording Finished!")

    # ------------------ ANALYSIS ------------------
    if len(emotion_timeline) > 0:

        st.markdown("---")
        st.subheader("📊 Results")

        emotion_counts = Counter(emotion_timeline)
        dominant_emotion = emotion_counts.most_common(1)[0][0]

        switches = sum(
            1 for i in range(1, len(emotion_timeline))
            if emotion_timeline[i] != emotion_timeline[i-1]
        )

        stability = 100 - (switches / len(emotion_timeline)) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("Dominant Emotion", dominant_emotion)
        c2.metric("Switches", switches)
        c3.metric("Stability", f"{round(stability,2)}%")

        # final graph
        numeric = [emotion_map[e] for e in emotion_timeline]

        fig, ax = plt.subplots()
        ax.plot(numeric, marker='o')
        ax.set_yticks(list(emotion_map.values()))
        ax.set_yticklabels(list(emotion_map.keys()))
        ax.set_title("Final Emotion Drift Timeline")
        ax.grid()

        st.pyplot(fig)

    else:
        st.warning("No emotions detected.")