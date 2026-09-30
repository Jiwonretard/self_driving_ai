`python -m drone_nav.train ...`가 완전 양자화된 `drone_nav_int8.tflite`를 이 폴더에 복사합니다.

모델 파일이 없으면 CMake가 의도적으로 빌드를 중단합니다. 임의 가중치 모델을 비행 장치에 넣지 않기 위한 안전장치입니다.

