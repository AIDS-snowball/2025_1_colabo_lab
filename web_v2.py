import os
import time
import base64
import streamlit as st
import cv2
import numpy as np
import time

from skeleton_matcher import match_user_to_video
from video_composer import composite_user_with_video

import torch
import mediapipe as mp
import math
import argparse
from mediapipe import solutions as mp_solutions
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Dropout

# ─── 설정 ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO = [
    #os.path.join(BASE_DIR, "assets", "video1.mp4"),
    #os.path.join(BASE_DIR, "assets", "video2.mp4"),
    #os.path.join(BASE_DIR, "assets", "video3.mp4")
    ['video/pokemon.mp4', 'pokemon', 0, 11, 'video/pokemon.png'],
    ['video/umpa.mp4', 'umpa', 0, 15, 'video/umpa.png'],
    ['video/next_level.mp4', 'next_level', 36, 61, 'video/next_level.png'],
    #'https://www.youtube.com/watch?v=9qyt9baCsQc',
    #'https://www.youtube.com/watch?v=9NdrlYmGbxA',
    #'https://www.youtube.com/watch?v=col1BYgfMjs',
]

st.set_page_config(page_title="🕺 동작 따라하기 챌린지", layout="wide")

st.markdown("""
<style>
  [data-testid="stVideo"] video,
  .stVideo > video {
    width: 100% !important;
    height: auto !important;
    aspect-ratio: 16 / 9;
    max-height: 65vh !important;
    object-fit: contain;
  }  
</style>
""", unsafe_allow_html=True)

# ─── 상태 초기화 ─────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = -1  # 시작화면
if "final_image" not in st.session_state:
    st.session_state.final_image = None
if "success_triggered" not in st.session_state:
    st.session_state.success_triggered = False
if "show_countdown" not in st.session_state:
    st.session_state.show_countdown = True

# ─── 시작 화면 ───────────────────────────────────────
def show_start_screen():
    st.markdown(
        "<h2 style='text-align: center; margin-top: 100px;'>아래 버튼을 눌러 게임을 시작하세요.</h2>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)

    button_css = """
        <style>
        .start-button {
            background: linear-gradient(90deg, #FF4B4B, #C62828);
            color: white;
            padding: 20px 60px;
            font-size: 24px;
            font-weight: bold;
            border: none;
            border-radius: 30px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: transform 0.3s ease, background 0.4s ease;
        }
        .start-button:hover {
            transform: scale(1.2);
            background: linear-gradient(90deg, #42A5F5, #1E88E5);
        }
        </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 게임 시작", key="start"):
            st.session_state.step = 0
            st.session_state.show_countdown = True
            st.rerun()

############## GCN 모델 정의 ##############
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)

############## 함수: landmark -> graph ##############
def create_graph_data(pre_lm_flat):
    edges = [
        (0,1),(1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),(8,9),
        (10,11),(11,12),(12,13),(13,14),
        (11,12),(11,23),(12,24),
        (23,24),(23,25),(24,26),
        (25,27),(26,28),
        #(27,29),(28,30),
        #(29,31),(30,32)
    ]
    edges += [(j, i) for i, j in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = torch.tensor(pre_lm_flat.reshape(-1, 4), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

############## 설정 및 초기화 ##############
threshold = 0.7

class_names = ['Chair', 'Cobra', 'Dog', 'next_level','pokemon', 'Tree','umpa', 'Warrior']
n_landmarks = 33
torso_size_multiplier = 2.5
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_channels=4, hidden_channels=64, num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('gcn_model.pth', map_location=device))
model.eval()

mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

############## 포즈 추론 함수 ##############
def process_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if not result.pose_landmarks:
        return img, 'No Pose'

    annotated_image = img.copy()
    lm_list = result.pose_landmarks.landmark

    # ✅ MediaPipe 스켈레톤 그리기
    mp_drawing.draw_landmarks(
        annotated_image,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # ✅ 정규화 및 그래프 입력 생성
    center_x = (lm_list[23].x + lm_list[24].x) * 0.5
    center_y = (lm_list[23].y + lm_list[24].y) * 0.5
    shoulders_x = (lm_list[11].x + lm_list[12].x) * 0.5
    shoulders_y = (lm_list[11].y + lm_list[12].y) * 0.5

    max_distance = max([
        math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2)
        for lm in lm_list
    ])
    torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
    norm_factor = max(torso_size * torso_size_multiplier, max_distance)

    pre_lm_flat = np.array([
        [(lm.x - center_x) / norm_factor,
         (lm.y - center_y) / norm_factor,
         lm.z / norm_factor,
         lm.visibility]
        for lm in lm_list
    ]).flatten()

    graph = create_graph_data(pre_lm_flat).to(device)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.batch)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        conf = np.max(probs)
        pred_label = class_names[np.argmax(probs)] if conf > threshold else 'Unknown Pose'

    # ✅ 예측 결과 텍스트 추가
    # cv2.putText(annotated_image, pred_label, (40, 80), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
    return annotated_image, pred_label

# ─── 웹캠 프레임 가져오기 ──────────────────────────────
def get_webcam_frame():
    webcam = cv2.VideoCapture(0)
    ret, frame = webcam.read()
    webcam.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

# ─── 웹캠 스켈레톤 가져오기 ──────────────────────────────
def extract_skeleton_only(frame):
    # MediaPipe Pose를 통해 스켈레톤만 시각화
    with mp.solutions.pose.Pose(static_image_mode=False) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
    return frame

# ─── 스테이지 화면 ───────────────────────────────────
def run_stage(video):
    import base64
    from pathlib import Path
    import cv2
    import time
    import streamlit as st

    video_path = video[0]
    current = st.session_state.step
    total = len(VIDEO)

    st.markdown(f"<h1 style='text-align:center; font-size:4.5rem;'>🕺 스테이지 {current + 1} 🎉</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:2.5rem;'>👀 영상을 보고 따라 해 보세요!!</p>", unsafe_allow_html=True)

    # 🔹 카운트다운
    if st.session_state.show_countdown:
        countdown = st.empty()
        for t in [3, 2, 1]:
            countdown.markdown(f"<h1 style='text-align:center'>{t}</h1>", unsafe_allow_html=True)
            time.sleep(1)
        countdown.markdown("<h1 style='text-align:center'>시작!</h1>", unsafe_allow_html=True)
        time.sleep(0.8)
        countdown.empty()
        st.session_state.show_countdown = False

    # 🔹 이전 / 다음 버튼
    col1, col2 = st.columns([1, 1])
    with col1:
        if current > 0:
            if st.button("⏪ 이전으로"):
                st.session_state.step -= 1
                st.session_state.show_countdown = True
                st.session_state.pose_matched = False
                st.session_state.replay_video = False
                st.rerun()
    with col2:
        if current < total:
            if st.button("⏩ 다음으로"):
                st.session_state.step += 1
                st.session_state.show_countdown = True
                st.session_state.pose_matched = False
                st.session_state.replay_video = False
                st.rerun()

    # 🔹 영상 + 웹캠
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<h4 style='text-align:center;'>📺 예제 영상</h4>", unsafe_allow_html=True)

        start_time = video[2]
        end_time = video[3]
        thumbnail_path = video[4]

        if st.session_state.get("replay_video", False):
            # ✅ 포즈 일치 시 영상 자동 재생
            video_file = Path(video_path)
            video_bytes = video_file.read_bytes()
            base64_video = base64.b64encode(video_bytes).decode()

            video_html = f"""
            <video id="myVideo" width="100%" controls autoplay>
                <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <script>
            const video = document.getElementById('myVideo');
            video.currentTime = {start_time};
            video.play();
            video.addEventListener('timeupdate', function () {{
                if (video.currentTime >= {end_time}) {{
                    video.pause();
                    // 자동 이동 제거
                }}
            }});
            </script>
            """
            st.components.v1.html(video_html, height=360)
        else:
            # ✅ 포즈 인식 전에는 썸네일만 보여줌
            st.image(thumbnail_path, use_container_width=True)

    with right:
        st.markdown("<h4 style='text-align:center;'>📷 실시간 웹캠</h4>", unsafe_allow_html=True)
        stframe = st.empty()
        status_placeholder = st.empty()

        if "pose_matched" not in st.session_state:
            st.session_state.pose_matched = False
        if not st.session_state.show_countdown: 
            webcam = cv2.VideoCapture(0)

            while True:
                ret_webcam, frame_webcam = webcam.read()
                if not ret_webcam or frame_webcam is None:
                    status_placeholder.warning("⚠️ 웹캠 프레임을 가져오지 못했습니다.")
                    break

                if not st.session_state.pose_matched:
                    # 🔹 포즈 인식 및 시각화
                    frame_out, pose_class = process_frame(frame_webcam)
                    frame_out = cv2.resize(frame_out, (640, 360))

                    if pose_class == video[1]:
                        st.session_state.pose_matched = True
                        st.session_state.replay_video = True
                        status_placeholder.success("🎯 너무 잘했어요!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        status_placeholder.error("🙅 동작이 다릅니다. 다시 시도해 보세요!")
                else:
                    # 🔹 스켈레톤만 계속 보여줌
                    frame_out = extract_skeleton_only(frame_webcam)
                    frame_out = cv2.resize(frame_out, (640, 360))
                    status_placeholder.success("🎯 너무 잘했어요!")

                stframe.image(frame_out, channels="BGR", use_container_width=True)
                time.sleep(1 / 30)

            webcam.release()

# ─── 마지막 결과 화면 ────────────────────────────────
def show_final_stage():
    st.success("🎉 모든 스테이지 완료!")
    img = composite_user_with_video()
    st.session_state.final_image = img
    st.image(img, caption="기념사진", use_container_width=True)

    img_bytes = cv2.imencode(".png", img)[1].tobytes()
    st.download_button("📥 사진 다운로드", data=img_bytes, file_name="final_photo.png", mime="image/png")

# ─── 라우팅 ───────────────────────────────────────────
if st.session_state.step == -1:
    show_start_screen()
elif st.session_state.step < len(VIDEO):
    run_stage(VIDEO[st.session_state.step])
else:
    show_final_stage()