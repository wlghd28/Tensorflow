/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/dtensor/mlir/create_dtensor_mlir_passes.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSOREMBEDDINGCHECKPOINT
#define GEN_PASS_DEF_DTENSOREMBEDDINGV2
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

struct DTensorEmbeddingV2
    : public impl::DTensorEmbeddingV2Base<DTensorEmbeddingV2> {
  void runOnOperation() override {}
};

struct DTensorEmbeddingCheckpoint
    : public impl::DTensorEmbeddingCheckpointBase<DTensorEmbeddingCheckpoint> {
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingPassV2() {
  return std::make_unique<DTensorEmbeddingV2>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingCheckpointPass() {
  return std::make_unique<DTensorEmbeddingCheckpoint>();
}

}  // namespace dtensor
}  // namespace tensorflow
