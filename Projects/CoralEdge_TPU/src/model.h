#pragma once

#define VERSION_C
//#define VERSION_CPP

#include <stdio.h>
#include <iostream>
#include <Windows.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"

// C version
#ifdef VERSION_C
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "edgetpu_c.h"
#endif

// C++ version
#ifdef VERSION_CPP
#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

using namespace edgetpu;
#endif

//#pragma comment (lib, "..\\lib\\tensorflowlite_c.lib")


using namespace cv;
using namespace std;

int create_model();
int drive_model();