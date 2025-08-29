#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/serialize.hpp"

#include <openvino/runtime/intel_npu/remote_properties.hpp>

#include <iostream>

int main(int, char*) {
try {
    std::cout << "starting strided sample\n";

    ov::Core core;

    const ov::Shape inputShape({1, 8, 12});
    const ov::Shape inputShapeSlice({1, 4, 6});
    const ov::element::Type_t elemType = ov::element::Type_t::u8;

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      7, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    uint8_t sanityData[] = {1, 2, 3, 4, 5, 6,
                            2, 2, 3, 4, 5, 6,
                            3, 2, 3, 4, 5, 6,
                            4, 2, 3, 4, 5, 6};

    ov::AnyMap main_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(data)}};

    ov::AnyMap sanity_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(sanityData)}};

    ov::Tensor input1(elemType, inputShape, data);
    ov::Tensor input_temp(elemType, inputShapeSlice);

    ov::Tensor sanity_input1(elemType, inputShapeSlice, sanityData);

    auto remoteCtx = core.get_default_context("NPU");
    auto zeroMainRemoteTensor = remoteCtx.create_tensor(elemType, inputShape, main_params);

    ov::RemoteTensor input1_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input2_roi1(zeroMainRemoteTensor, {0, 0, 0}, {1, 4, 6});
    ov::RemoteTensor input1_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input2_roi2(zeroMainRemoteTensor, {0, 4, 0}, {1, 8, 6});
    ov::RemoteTensor input1_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input2_roi3(zeroMainRemoteTensor, {0, 0, 6}, {1, 4, 12});
    ov::RemoteTensor input1_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});
    ov::RemoteTensor input2_roi4(zeroMainRemoteTensor, {0, 4, 6}, {1, 8, 12});

    auto sanityInputRemoteTensor = remoteCtx.create_tensor(elemType, inputShapeSlice, sanity_params);


    auto sanityCheckParam1 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto sanityCheckParam2 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto sanityCheckMultiply = std::make_shared<ov::op::v1::Multiply>(sanityCheckParam1, sanityCheckParam2);
    const auto sanityCheckResults = ov::ResultVector{std::make_shared<ov::opset1::Result>(sanityCheckMultiply->output(0))};
    auto sanityCheckModel = std::make_shared<ov::Model>(sanityCheckResults, ov::ParameterVector{sanityCheckParam1, sanityCheckParam2}, "EltwiseMultiply");
    ov::CompiledModel compiled_sanitycheck_model = core.compile_model(sanityCheckModel, "NPU");
    ov::InferRequest infer_sanitycheck_request = compiled_sanitycheck_model.create_infer_request();
    infer_sanitycheck_request.set_input_tensor(0, sanityInputRemoteTensor);
    infer_sanitycheck_request.set_input_tensor(1, sanityInputRemoteTensor);
    infer_sanitycheck_request.infer();
    auto outputSanityCheck = infer_sanitycheck_request.get_output_tensor(0);
    auto outSanityData = outputSanityCheck.data<uint8_t>();
    std::cout << "printing sanity output\n";
    size_t dataIdx1 = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(outSanityData[dataIdx1]) << " ";
            dataIdx1++;
        }
        std::cout << "\n";
    }

    auto param1 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(elemType, inputShapeSlice);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(param1, param2);
    multiply->get_output_tensor(0).set_names({"Multiply_Result"});

    const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(multiply->output(0))};
    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{param1, param2}, "EltwiseMultiply");

    ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    infer_request.set_input_tensor(0, input1_roi1);
    infer_request.set_input_tensor(1, input2_roi1);
    infer_request.infer();

    auto outputTensor = infer_request.get_output_tensor(0);
    auto outData = outputTensor.data<uint8_t>();

    size_t dataIdx = 0;
    uint8_t* inData;

    std::cout << "printing tile1\n";
    std::cout << "input\n";

    infer_request.get_input_tensor(0).copy_to(input_temp);
    inData = input_temp.data<uint8_t>();
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(inData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    std::cout << "printing output\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(outData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    infer_request.set_input_tensor(0, input1_roi2);
    infer_request.set_input_tensor(1, input2_roi2);
    infer_request.infer();

    std::cout << "printing tile2\n";
    std::cout << "input\n";

    infer_request.get_input_tensor(0).copy_to(input_temp);
    inData = input_temp.data<uint8_t>();
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(inData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    std::cout << "printing output\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(outData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    infer_request.set_input_tensor(0, input1_roi3);
    infer_request.set_input_tensor(1, input2_roi3);
    infer_request.infer();

    std::cout << "printing tile3\n";
    std::cout << "input\n";

    infer_request.get_input_tensor(0).copy_to(input_temp);
    inData = input_temp.data<uint8_t>();
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(inData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    std::cout << "printing output\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(outData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    infer_request.set_input_tensor(0, input1_roi4);
    infer_request.set_input_tensor(1, input2_roi4);
    infer_request.infer();

    infer_request.get_input_tensor(0).copy_to(input_temp);
    inData = input_temp.data<uint8_t>();
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(inData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    std::cout << "printing output\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 4; idx++) {
        for (size_t idx2 = 0; idx2 < 6; idx2++) {
            std::cout << static_cast<int>(outData[dataIdx]) << " ";
            dataIdx++;
        }
        std::cout << "\n";
    }

    std::cout << "MaxPool test\n";

    const ov::Shape maxPoolInputShape({1, 16, 108, 1280});
    const ov::Shape maxPoolSliceInputShape({1, 16, 36, 1280});
    const ov::element::Type_t maxPoolElemType = ov::element::Type_t::f32;

    float* maxPoolData = new float[2211840];

    for (int idx = 0; idx < 2211840; idx++) {
        maxPoolData[idx] = (idx % 256);
    }

    ov::AnyMap max_pool_params = {{ov::intel_npu::mem_type.name(), ov::intel_npu::MemType::L0_INTERNAL_BUF},
                         {ov::intel_npu::tensor_type.name(), {ov::intel_npu::TensorType::INPUT}},
                         {ov::intel_npu::mem_handle.name(), reinterpret_cast<void*>(maxPoolData)}};

    auto zeroMaxPoolMainTensor = remoteCtx.create_tensor(maxPoolElemType, maxPoolInputShape, max_pool_params);

    ov::RemoteTensor max_pool_roi1(zeroMaxPoolMainTensor, {0, 0, 0, 0}, {1, 16, 36, 1280});
    ov::RemoteTensor max_pool_roi2(zeroMaxPoolMainTensor, {0, 0, 36, 0}, {1, 16, 72, 1280});
    ov::RemoteTensor max_pool_roi3(zeroMaxPoolMainTensor, {0, 0, 72, 0}, {1, 16, 108, 1280});

    auto max_pool_param = std::make_shared<ov::op::v0::Parameter>(maxPoolElemType, maxPoolSliceInputShape);

    auto max_pool = std::make_shared<ov::op::v1::MaxPool>(max_pool_param, ov::Strides{2, 2}, 
                                                            ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{4, 4});
    max_pool->get_output_tensor(0).set_names({"MaxPool_Results"});

    const auto max_pool_results = ov::ResultVector{std::make_shared<ov::opset1::Result>(max_pool->output(0))};
    auto max_pool_model = std::make_shared<ov::Model>(max_pool_results, ov::ParameterVector{max_pool_param}, "MaxPool");

    ov::serialize(model, "maxpool.xml", "maxpool.bin");

    ov::CompiledModel compiled_max_pool = core.compile_model(max_pool_model, "NPU");
    ov::InferRequest max_pool_infer_request = compiled_max_pool.create_infer_request();

    max_pool_infer_request.set_input_tensor(0, max_pool_roi1);
    max_pool_infer_request.infer();

    auto outputMaxPoolTensor = max_pool_infer_request.get_output_tensor(0);
    auto outMaxPoolData = outputMaxPoolTensor.data<float>();

    std::cout << "printing max pool output 1\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    max_pool_infer_request.set_input_tensor(0, max_pool_roi2);
    max_pool_infer_request.infer();

    std::cout << "printing max pool output 2\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    max_pool_infer_request.set_input_tensor(0, max_pool_roi3);
    max_pool_infer_request.infer();

    std::cout << "printing max pool output 3\n";
    dataIdx = 0;
    for (size_t idx = 0; idx < 16; idx++) {
        for (size_t idx2 = 0; idx2 < 36; idx2++) {
            for (size_t idx3 = 0; idx3 < 1280; idx3++) {
                std::cout << static_cast<int>(outMaxPoolData[dataIdx]) << " ";
                dataIdx++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


} catch (const std::exception& ex) {
    std::cout << ex.what() << std::endl;
    return EXIT_FAILURE;
}
    return EXIT_SUCCESS;
}