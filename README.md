# HandPoseVirtualMouse

# CNN_HandRecognition.py는 Tensorflow CNN을 이용하여 5가지 손 포즈를 학습시켜 학습 모델을 얻는 코드
# hand_recognition_cnn.pb는 위의 코드를 실행하여 얻은 학습 모델
# opencv_basic_code_console.cpp는 손영역을 검출하고 학습모델을 통해 예측한 값에 따라 다른 마우스 이벤트를 발생시켜 핸드 가상 마우스 구현하는 코드
# opencv_basic_code_console_video_to_frame_code.cpp는 5가지 손 포즈를 영상으로 촬영 후 프레임마다 캡처하여 손영역만을 검출해 잘라 학습데이터로 저장하는 코드
