#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"
#include "c_api.h"
#include "c_api_experimental.h"
#include "common.h"
#include "edgetpu.h"
#include "edgetpu_c.h"

//#pragma comment (lib, "..\\lib\\tensorflowlite_c.lib")

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{

    //absl::ParseCommandLine(argc, argv);

    //// Load the model.
    //const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
    //auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
    //    ? coral::GetEdgeTpuContextOrDie()
    //    : nullptr;
    //auto interpreter = coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
    //CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    //// Read the image to input tensor.
    //auto input = coral::MutableTensorData<char>(*interpreter->input_tensor(0));
    //coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path), input.data(), input.size());
    //CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

    //// Read the label file.
    //auto labels = coral::ReadLabelFile(absl::GetFlag(FLAGS_labels_path));

    //for (auto result : coral::GetClassificationResults(*interpreter, 0.0f, /*top_k=*/3)) {
    //    std::cout << "---------------------------" << std::endl;
    //    std::cout << labels[result.id] << std::endl;
    //    std::cout << "Score: " << result.score << std::endl;
    //}

    printf("%s\n", TfLiteVersion());
	return 0;
}