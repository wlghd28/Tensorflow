#include "model.h"

const std::vector<std::string> labels = {
  "???", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "???", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear",
  "zebra", "giraffe", "???", "backpack", "umbrella",
  "???", "???", "handbag", "tie", "suitcase",
  "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "???", "wine glass", "cup", "fork",
  "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza","donut","cake","chair","couch",
  "potted plant","bed","???","dining table","???",
  "???","toilet","???","tv","laptop",
  "mouse","remote","keyboard","cell phone","microwave",
  "oven","toaster","sink","refrigerator","???",
  "book","clock","vase","scissors","teddy bear",
  "hair drier","toothbrush"
};

int create_model()
{
#ifdef VERSION_CPP

#endif

#ifdef VERSION_C


#endif

	return 0;
}


int drive_model()
{
    // Load Test orgImage
    Mat orgImage = cv::imread(std::string("./image/airplane.jpg"));
    Mat resizeImage;
    cv::resize(orgImage, resizeImage, { 300, 300 });

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

    //assert(num_devices > 0);
    if (num_devices <= 0) return -1;
    const auto& device = devices.get()[0];

    // 4. Modify interpreter with the delegate.
    auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    TfLiteStatus tfRet = TfLiteInterpreterModifyGraphWithDelegate(interpreter, delegate);

    // 5. Prepare input tensors and run inference.
    TfLiteInterpreterAllocateTensors(interpreter);
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    TfLiteTensorCopyFromBuffer(input_tensor, resizeImage.ptr(0), TfLiteTensorByteSize(input_tensor));
    //.... (Prepare input tensors)

    //printf("%d %d\n", TfLiteTensorByteSize(input_tensor), resizeImage.size().width * resizeImage.size().height * sizeof(unsigned char));
    // Extract the output tensor data.
    TfLiteInterpreterInvoke(interpreter);
    //const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    //TfLiteTensorCopyToBuffer(output_tensor, output, TfLiteTensorByteSize(output_tensor));
    const TfLiteTensor* rects_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    const TfLiteTensor* classes_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 1);
    const TfLiteTensor* scores_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 2);
    const TfLiteTensor* numDetect_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 3);

    float* rects = (float*)malloc(TfLiteTensorByteSize(rects_tensor));
    if (rects == NULL) return -1;
    float* classes = (float*)malloc(TfLiteTensorByteSize(classes_tensor));
    if (classes == NULL) return -1;
    float* scores = (float*)malloc(TfLiteTensorByteSize(scores_tensor));
    if (scores == NULL) return -1;
    float* numDetect = (float*)malloc(TfLiteTensorByteSize(numDetect_tensor));
    if (numDetect == NULL) return -1;

    TfLiteTensorCopyToBuffer(rects_tensor, rects, TfLiteTensorByteSize(rects_tensor));
    TfLiteTensorCopyToBuffer(classes_tensor, classes, TfLiteTensorByteSize(classes_tensor));
    TfLiteTensorCopyToBuffer(scores_tensor, scores, TfLiteTensorByteSize(scores_tensor));
    TfLiteTensorCopyToBuffer(numDetect_tensor, numDetect, TfLiteTensorByteSize(numDetect_tensor));
    //.... (Retrieve the result from output tensors)

    // OpenCV
    //printf("%d %d %d %d\n", (unsigned int)_msize(rects), (unsigned int)_msize(classes), (unsigned int)_msize(scores), (unsigned int)_msize(numDetect));

    const auto size = rects_tensor->dims->size;

    for (int i = 0; i < size; ++i) {
        cv::Point2d tr{ (double)rects[i * 4 + 1] * orgImage.cols, (double)rects[i * 4] * orgImage.rows };

        cv::rectangle(orgImage,
            cv::Point2d{ (double)rects[i * 4 + 1] * (double)orgImage.cols, (double)rects[i * 4] * (double)orgImage.rows },
            cv::Point2d{ (double)rects[i * 4 + 3] * (double)orgImage.cols, (double)rects[i * 4 + 2] * (double)orgImage.rows },
            { 255,0,0 }, 2);

        char buf[10];
        std::sprintf(buf, "(%.1f%%)", (double)scores[i] * 100);

        cv::putText(orgImage,
            labels[std::floor(classes[i] + 1.5)] + std::string(buf),
            cv::Point2d{ (double)rects[i * 4 + 1] * (double)orgImage.cols, (double)rects[i * 4] * (double)orgImage.rows - 4 },
            cv::FONT_ITALIC, 0.6, { 0,0,0 }, 2);
    }


    // Dispose of the model and interpreter objects.
    edgetpu_free_delegate(delegate);
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    free(rects);
    free(classes);
    free(scores);
    free(numDetect);
#endif

    imwrite("Result.jpg", orgImage);
    imshow("Result", orgImage);
    waitKey(0);

    return 0;
}