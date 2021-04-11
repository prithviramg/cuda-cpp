#include "dataLoader.h"
#include "network.h"
#include "layer.h"

#include <iomanip>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    /* network configuration */
    int train_batchSize = 256;
    int train_steps = 1600;
    int test_batchSize = 10;
    int test_steps = 1000;
    

    double learningRate = 0.02f;
    double learningRateDecay = 0.00005f;
    int monitoringStep = 200;

    bool loadPretrain = false;
    bool file_save = false;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;
    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    cudl::MNIST train_dataGenerator = cudl::MNIST("./dataset");
    train_dataGenerator.train(train_batchSize, true);

    // step 2. model initialization
    cudl::Network model;
    model.addLayer(new cudl::Dense("dense1", 500));
    model.addLayer(new cudl::Activation("relu", CUDNN_ACTIVATION_RELU));
    model.addLayer(new cudl::Dense("dense2", 10));
    model.addLayer(new cudl::Softmax("softmax"));
    model.cuda();

    if (loadPretrain)
        model.loadPretrain();
    model.train();

    // start Nsight System profile
    cudaProfilerStart();

    // step 3. train
    int step = 0;
    cudl::Blob<float> *train_data = train_dataGenerator.get_data();
    cudl::Blob<float> *train_target = train_dataGenerator.get_target();
    train_dataGenerator.get_batch();
    int tp_count = 0;
    while (step < train_steps)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        train_data->to(cudl::cuda);
        train_target->to(cudl::cuda);
        
        // forward
        model.forward(train_data);
        tp_count += model.get_accuracy(train_target);

        // back-propagation
        model.backward(train_target);

        // update parameter
        // we will use learning rate decay to the learning rate
        learningRate *= 1.f / (1.f + learningRateDecay * step);
        model.update(learningRate);

        // fetch next data
        step = train_dataGenerator.next();

        // nvtx profiling end
        nvtxRangePop();

        // calculation softmax loss
        if (step % monitoringStep == 0)
        {
            float loss = model.loss(train_target);
            float accuracy =  100.f * tp_count / monitoringStep / train_batchSize;
            
            std::cout << "step: " << std::right << std::setw(4) << step << \
                         ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << \
                         ", accuracy: " << accuracy << "%" << std::endl;

            tp_count = 0;
        }
    }

    // trained parameter save
    if (file_save)
        model.write_file();

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    cudl::MNIST test_data_loader = cudl::MNIST("./dataset");
    test_data_loader.test(test_batchSize);

    // step 2. model initialization
    model.test();
    
    // step 3. iterates the testing loop
    cudl::Blob<float> *test_data = test_data_loader.get_data();
    cudl::Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    tp_count = 0;
    step = 0;
    while (step < test_steps)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
        test_data->to(cudl::cuda);
        test_target->to(cudl::cuda);

        // forward
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }

    // stop Nsight System profiling
    cudaProfilerStop();

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / test_steps / test_batchSize;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    // Good bye
    std::cout << "Done." << std::endl;

    return 0;
}
