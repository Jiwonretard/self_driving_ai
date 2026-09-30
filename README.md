# ESP32-S3 카메라 드론 경량 비전 모델

카메라 한 장으로 다음 비행 방향을 분류하는 초경량 TinyML 프로젝트입니다.

- 입력: `96 x 96 x 3` RGB, `uint8`
- 출력: `forward`, `left`, `right`, `stop`
- 모델: depthwise-separable CNN, 완전 정수 양자화
- PC 검증: MediaPipe Tasks Image Classifier
- 보드 추론: Espressif `esp-tflite-micro` + ESP-NN
- 안전 동작: 신뢰도가 임계값보다 낮거나 추론에 실패하면 항상 `stop`

> 이 모델은 비행제어기를 대신하지 않습니다. 자세 안정화와 모터 믹싱은 PX4/ArduPilot 같은 별도 비행제어기가 담당해야 합니다. 첫 시험은 프로펠러를 제거하고 진행하세요.

## 왜 ESP32-S3인가

기본 타깃은 PSRAM이 있는 ESP32-S3 카메라 보드입니다. 같은 모델을 기존 ESP32-CAM에서도 빌드할 수 있지만 추론 지연과 메모리 여유가 크게 나빠집니다. Espressif의 공식 벤치마크도 S3에서 ESP-NN 최적화 효과가 훨씬 큽니다.

MediaPipe 전체 런타임을 MCU에 올리는 방식은 아닙니다. 학습된 TFLite 모델을 MediaPipe Tasks로 검증하고, MCU에서는 같은 모델을 TFLite Micro로 실행합니다. 이것이 메모리가 작은 ESP32 계열에서 현실적인 배포 경로입니다.

## 1. 환경 준비

Python 3.10 또는 3.11을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

## 2. 데이터 구성

각 사진은 **그 사진을 본 순간 취해야 할 안전한 행동** 폴더에 넣습니다.

```text
dataset/
  train/
    forward/*.jpg
    left/*.jpg
    right/*.jpg
    stop/*.jpg
  val/
    forward/*.jpg
    left/*.jpg
    right/*.jpg
    stop/*.jpg
  test/
    forward/*.jpg
    left/*.jpg
    right/*.jpg
    stop/*.jpg
```

클래스별 최소 1,000장부터 시작하고, 실제 탑재 위치·렌즈·해상도·조명에서 수집하세요. 사람, 동물, 알 수 없는 장애물, 흐림, 역광은 `stop`에 충분히 포함해야 합니다. 연속 프레임을 임의로 섞어 분할하면 데이터 누수가 생기므로 촬영 세션 단위로 train/val/test를 나누세요.

데이터 형식을 먼저 검사할 수 있습니다.

```bash
python tools/check_dataset.py --data dataset
```

## 3. 학습 및 INT8 변환

```bash
python -m drone_nav.train \
  --data dataset \
  --epochs 40 \
  --output artifacts
```

학습 스크립트는 다음을 수행합니다.

1. 96×96 RGB 입력과 작은 separable CNN 학습
2. validation loss 기준 최적 가중치 복원
3. train 이미지로 representative calibration
4. built-in 정수 연산만 허용한 UINT8 TFLite 변환
5. TensorFlow 모델과 TFLite 모델의 test 정확도 비교
6. `firmware/main/model/drone_nav_int8.tflite`로 모델 복사

모델 크기나 정확도 기준을 못 맞추면 명령이 실패합니다. 기본 제한은 180 KiB, float 대비 양자화 정확도 하락 5%p 이하입니다.

## 4. MediaPipe에서 검증

```bash
python tools/validate_mediapipe.py \
  --model artifacts/drone_nav_int8.tflite \
  --labels artifacts/labels.txt \
  --image path/to/frame.jpg
```

## 5. ESP32-S3 펌웨어

ESP-IDF 5.1 이상을 설치한 뒤:

```bash
cd firmware
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py --port /dev/ttyUSB0 flash monitor
```

`menuconfig > Drone navigation`에서 카메라 핀, UART 핀, 신뢰도 임계값을 실제 보드에 맞추세요. 기본 핀은 흔히 쓰이는 Freenove ESP32-S3 WROOM CAM 계열 예시일 뿐이므로 회로도 확인이 필수입니다.

비행제어기 UART에는 아래 한 줄이 주기적으로 전송됩니다.

```text
NAV,42,stop,0.812,37
```

필드는 `sequence, command, confidence, inference_ms`입니다. 비행제어기는 패킷이 끊기거나 `stop`을 받으면 정지/호버/수동 전환하도록 별도 fail-safe를 구현해야 합니다.

## 빠른 코드 테스트

```bash
pytest -q
c++ -std=c++17 -Wall -Wextra -Werror \
  -Ifirmware/main tests/decision_filter_test.cc \
  firmware/main/decision_filter.cc -o /tmp/decision_filter_test
/tmp/decision_filter_test
```

## 모델 승인 체크리스트

- test split이 촬영 세션 기준으로 완전히 분리됨
- `stop` recall이 목표 이상(권장 0.98 이상)
- 역광, 야간, motion blur, 부분 가림에서 검증함
- 보드 실측 추론 주기가 비행제어기의 요구보다 충분히 빠름
- UART 끊김, 카메라 실패, 모델 실패 시 fail-safe를 확인함
- 프로펠러 제거 → 고정 리그 → 보호망 내부 순으로 시험함
