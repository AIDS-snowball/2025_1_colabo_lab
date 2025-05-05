import numpy as np

def composite_user_with_video() -> np.ndarray:
    """
    사용자 segment 결과와 배경 영상 합성하여 최종 이미지 생성

    Returns:
        np.ndarray: 최종 이미지 (RGB 형태)
    """
    # TODO: 사용자 세그멘테이션 + 영상 배경 합성
    # 임시로 흰색 배경 이미지 반환
    return np.ones((480, 640, 3), dtype=np.uint8) * 255
