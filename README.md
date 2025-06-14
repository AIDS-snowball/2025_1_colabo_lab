# 2025_1_colabo_lab

2025년 4~6월에 진행한 SW/AI 테마파크 내용입니다.

## 실행
```
streamlit run app.py
```

## 개발 배경

프로그램 체험자가 4,5세부터 초등학생인 점을 감안하여, 간단한 동작만으로도 프로그램을 즐길 수 있는 방향으로 구성함. 

그 결과 영상 분야로 결정하게 되었고 채험자가 쉽게 접할 수 있도록 동작을 이용한 체험 프로그램을 기획함.

본 프로그램은 스켈레톤 데이터의 분류 작업을 통해 실시간으로 행하는 동작이 목표 동작과 일치하는지를 판별함.

## 프로그램 동작 과정
* 체험자의 동작과 화면속의 동작이 일치하는지 판별
* 동작이 일치하면 점수를 얻고 다음 동작으로 넘어감
* 동작이 일치하거나 제한시간이 넘어가면 영상이 재생

## 모델 학습과정
### 데이터 수집
* 먼저 프로그램에 사용할 콘텐츠를 움파룸파, 포켓몬스터 노래, aespa의 next level로 잡고, 데이터 수집을 진행.
  * 각 2만장씩 총 6만장 수집.
* 동작은 인식하기위한 모델은 google의 mediapipe를 활용.
* mediapipe를 활용해 각 데이터의 포즈 데이터 추출
* 오픈 소스 포즈 데이터를 통해 학습에 활용할 데이터 보충 이를 통해 모델의 일반화 성능 및 안정성 강화

### 모델 학습
포즈의 일치/불일치를 판별하기 위해 다음과 같은 방법 시도. 
- Euclidean Distance
- Neural network
- GCN with BCE
- siamese netwrok with GCN
- GCN with focal loss
- spatial-temporal graph convolution - st-gcn
- 2s-agcn
- poseformer mix

추론 결과와 일반화 성능을 중점으로 1차적으로 테스트 진행.

1차 테스트에서 선정한 모델은 추가적으로 데이터를 보충해가며 필드 테스트의 안정성 테스트 및 강화.

모든 과정에서 하이퍼파라미터 수정.

추론 결과와 일반화 성능 그리고 필드 테스트의 안정성을 고려하여 GCN with focal loss 모델로 결정

> go to [model_train](./model_train)
> * data.csv: what we made
> * GCN.ipynb: model(GCN) training code

### 웹사이트 구현
streamlit를 활용해 익숙한 파이썬을 통해 웹사이트 제작.

어린 체험자들이 쉽게 프로그램을 체험할 수 있도록 시각적 요소를 중시 함.

- 첫 페이지에 웹캠을 보여주면서 어떤 콘텐츠인지 가늠할 수 있도록 함.
- 동작이 일치하면 풍선을 띄어줌

## 프로그램 소개
1. 시작 페이지: 체험자에 맞게 카메라 세팅.

![image](https://github.com/user-attachments/assets/3ca70fc6-993d-4046-a54c-891d005f6b84)

2. 게임 시작을 클릭하면 맞춰야하는 동작이 나오고 동작이 일치하는지 판별.
  - 동작이 일치하면 풍선이 올라오고 다음 스테이지
  - 제한시간이 지나거나 동작이 일치하면 동영상 재생
    

![image](https://github.com/user-attachments/assets/a352c6c6-d4e7-4e72-a934-090eb93413a8)


![image](https://github.com/user-attachments/assets/b23ff13e-d547-4fba-b3ed-99e1331ecb89)

3. 2를 반복
