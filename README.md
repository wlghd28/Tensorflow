# 개발 툴
- Visual Studio 2019

# Tensorflowlite 빌드
- 아래 명령문 입력 (git bash 활용)
    1. cmake ../tensorflow_src/tensorflow/lite/c -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True
    2. cmake --build . -j
- 명령문 입력 후 tensorflowlite_c.lib, tensorflowlite_c.dll 파일을 프로젝트에 링크시킨다.

# libedgetpu 빌드
- edgetpu.dll.if.lib, edgetpu.dll 파일을 프로젝트에 링크시킨다.


# 참고자료
- (https://www.tensorflow.org/lite/guide/build_cmake?hl=ko)
- (https://coral.ai/docs/edgetpu/tflite-cpp/#run-an-inference-with-the-libcoral-api)