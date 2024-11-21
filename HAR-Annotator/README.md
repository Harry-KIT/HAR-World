
## YOLOv7-Pose Annotation Tool
## 소개
이 저장소에는 YOLOv7-Pose 사전 학습된 모델을 사용하여 액션 비디오에 주석을 달기 위한 도구가 들어 있습니다. 이 도구를 사용하면 비디오의 주요 지점에 자동으로 주석을 달 수 있어 인간 액션을 분석하고 시각화하기가 더 쉬워집니다.

## 🚀 실행
annotator.py에서 매개변수를 수정할 수 있습니다.
```
python annotator.py
```
## :white_square_button: 데모
visualize.py를 사용하여 결과를 시각화할 수 있습니다.
```
python visualize.py
```

걷기|서있기|달리기
--|--|--
<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/walking.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/standing.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/running.gif" width="300" />

앉기|쪼그리고 앉기|도약하기
--|--|--
<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/sit_down.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/squating.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/jumping.gif" width="300" />

손을 흔들기|발길질하기|펀칭
--|--|--
<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/hand_waving.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/kicking.gif" width="300" />|<img src="https://github.com/Harry-KIT/HAR-World/blob/main/HAR-Annotator/assets/punching.gif" width="300" />

## :pushpin: Pretrained Models
GitHub Link: [yolov7-w6-pose](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

## :pushpin: Dataset
Google Drive: [Action Data Samples](https://drive.google.com/file/d/1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9/view)

## 👀 Contact Me
If you have any questions, please feel free to email me at [ai.devveloper@gmail.com](ai.devveloper@gmail.com).
