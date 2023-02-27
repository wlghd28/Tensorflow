#define VERSION_C
//#define VERSION_CPP

#include <stdio.h>
#include <iostream>
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



int main(int argc, char* argv[])
{
#ifdef VERSION_CPP
    // Sets up the tpu_context.
    auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();

    // Build the interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // Load the model
    std::string model_file_name = std::string("model/test.tflite");
    auto model = tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());

    // Registers Edge TPU custom op handler with Tflite resolver.
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Binds a context with a specific interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context.get());

    // Note that all edge TPU context set ups should be done before this
    // function is called.
    interpreter->AllocateTensors();
    //.... (Prepare input tensors)
    interpreter->Invoke();
    //.... (retrieving the result from output tensors)

    // Releases interpreter instance to free up resources associated with
    // this custom op.
    interpreter.reset();

    // Closes the edge TPU.
    tpu_context.reset();
#endif

#ifdef VERSION_C
    // Create the model and interpreter options.
    // 1. Create tflite::FlatBufferModel which may contain edge TPU custom op.
    std::string model_file_name = std::string("../model/ssd_mobilenet_v1_1_metadata_1.tflite");
    TfLiteModel* model = TfLiteModelCreateFromFile(model_file_name.c_str());
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 2);

    // 2. Create tflite::Interpreter.
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    // 3. Enumerate edge TPU devices.
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    assert(num_devices > 0);
    const auto& device = devices.get()[0];

    // 4. Modify interpreter with the delegate.
    auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    TfLiteStatus ret = TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate);

    // 5. Prepare input tensors and run inference.
    TfLiteInterpreterAllocateTensors(interpreter);
    //TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    //TfLiteTensorCopyFromBuffer(input_tensor, input.data(), input.size() * sizeof(float));
    //.... (Prepare input tensors)

    // Extract the output tensor data.
    TfLiteInterpreterInvoke(interpreter);
    //const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    //TfLiteTensorCopyToBuffer(output_tensor, output.data(), output.size() * sizeof(float));
    //.... (Retrieve the result from output tensors)

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
  
#endif
  
    printf("TfLiteVersion : %s\n", TfLiteVersion());

	return 0;
}