#include <iostream>

#include "DL/Model.h"

#include "util/Print.h"
#include "util/PrintModel.h"

int main()
{
    constexpr size_t EPOCH = 256;

    ml::BasicModel x(
        ml::Metaparams(
            {2, 8, 16, 16, 8, 4, 1},
            {
                ml::ActFunc::None,
                ml::ActFunc::ReLU,
                ml::ActFunc::ReLU,
                ml::ActFunc::ReLU,
                ml::ActFunc::None,
                ml::ActFunc::Sigmoid,
            }
        )
    );

    //ut::printBasicModelData(x);

    ml::Batch input = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
    };

    ml::Batch target = {
        {0.0f},
        {1.0f},
        {1.0f},
        {1.0f},
    };

    // {
    //     std::vector<float> output = x.forward({1.0f, 1.0f});
    //     std::cout << "INITIAL PARAMETERS:\n";
    //     std::cout << "Cost  :" << ml::cost::mean(output.data(), target[3], output.size(), ml::cost::func::BinaryCrossEntropy);
    //     std::cout << "\nOutput: ";
    //     ut::printArray(output.data(), output.size());
    //     std::cout << "\n\n";
    // }

    for (size_t i = 0; i < EPOCH; i++)
    {
        x.update(input, target, ml::cost::deriv::BinaryCrossEntropy, 0.1f);

        // std::vector<float> output = x.forward({1.0f, 1.0f});
        // std::cout << "ITERATION [" << (i + 1) << "]:\n";
        // std::cout << "Cost  :" << ml::cost::mean(output.data(), target[3], output.size(), ml::cost::func::BinaryCrossEntropy);
        // std::cout << "\nOutput: ";
        // ut::printArray(output.data(), output.size());
        // std::cout << "\n\n";
    }

    std::vector<float> output = x.forward({0.0f, 0.0f});
    std::cout << "[[ RESULTS ]]"
    << "\n\nDATA [1]:\n"
    << "INPUTS: {0.0f, 0.0f}\n"
    << "OUTPUTS: ";
    ut::printArray(output.data(), output.size());

    std::cout
    << "\n\nDATA [1]:\n"
    << "INPUTS: {1.0f, 0.0f}\n"
    << "OUTPUTS: ";
    output = x.forward({1.0f, 0.0f});
    ut::printArray(output.data(), output.size());

    std::cout
    << "\n\nDATA [1]:\n"
    << "INPUTS: {0.0f, 1.0f}\n"
    << "OUTPUTS: ";
    output = x.forward({0.0f, 1.0f});
    ut::printArray(output.data(), output.size());

    std::cout
    << "\n\nDATA [1]:\n"
    << "INPUTS: {1.0f, 1.0f}\n"
    << "OUTPUTS: ";
    output = x.forward({1.0f, 1.0f});
    ut::printArray(output.data(), output.size());

    return 0;
}
