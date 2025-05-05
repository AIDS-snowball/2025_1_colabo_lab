import streamlit as st
import cv2
import numpy as np

from skeleton_matcher import match_user_to_video
from video_composer import composite_user_with_video

VIDEO_PATHS = ["assets/video1.mp4", "assets/video2.mp4", "assets/video3.mp4"]

st.set_page_config(page_title="동작 따라하기 게임", layout="centered")
st.title("🕺 동작 따라하기 챌린지")

if "step" not in st.session_state:
    st.session_state.step = 0
if "final_image" not in st.session_state:
    st.session_state.final_image = None

def run_stage(video_path):
    st.subheader(f"스테이지 {st.session_state.step + 1}")
    st.video(video_path)
    st.info("영상을 보고 동작을 따라하세요!")

    if st.button("동작 확인하기"):
        with st.spinner("분석 중..."):
            matched = match_user_to_video(video_path)
            if matched:
                st.success("🎯 일치했습니다!")
                st.session_state.step += 1
            else:
                st.error("🙅 동작이 다릅니다. 다시 시도하세요!")

def show_final_stage():
    st.success("🎉 모든 스테이지 완료!")
    img = composite_user_with_video()
    st.session_state.final_image = img
    st.image(img, caption="기념사진")

    img_bytes = cv2.imencode(".png", img)[1].tobytes()
    st.download_button("📥 사진 다운로드", data=img_bytes, file_name="final_photo.png", mime="image/png")

if st.session_state.step < len(VIDEO_PATHS):
    run_stage(VIDEO_PATHS[st.session_state.step])
else:
    show_final_stage()
