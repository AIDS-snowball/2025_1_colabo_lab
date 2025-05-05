def match_user_to_video(video_path: str) -> bool:
    """
    사용자 실시간 포즈와 영상 포즈를 비교해 유사 여부 판단

    Args:
        video_path (str): 현재 비교 중인 영상 경로

    Returns:
        bool: True (일치) or False (불일치)
    """
    # TODO: 사용자의 skeleton 추출 후 video_path와 비교
    # 예: MediaPipe, OpenPose 등 사용
    return True  # 임시로 항상 일치 처리
