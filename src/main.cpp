#include <iostream>

#include "DL/Cost.h"
#include "DL/Propagation.h"

#include "util/Print.h"

int main()
{
    // Training a model of only 2 layers: input layer (2 nodes) and output layer (2 nodes)

    std::system("cls");

    constexpr size_t EPOCH = 64;

    ml::fVector parameters = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };

    ml::ParamMatrix matrix(parameters.data(), 2, 2, ml::ActFunc::None);

    ml::Batch input = {
        {1.0f, 1.0f}
    };

    ml::Batch target = {
        {1.0f, 1.0f}
    };

    for (size_t i = 0; i < EPOCH; i++)
    {
        ml::Batch output = ml::propagation::forward(matrix, input);
        ml::Batch dOutput(1, 2);

        ml::cost::apply(dOutput.data(), output.data(), target.data(), 2, ml::cost::deriv::MeanSquaredError);

        matrix.update(input, output, dOutput, 0.1f);

        // std::cout << '\n';
        // ut::printParamMatrix(parameters.data(), 2, 2);
        // std::cout << '\n';
        
        std::cout
        << "[" << i << "]:\n"
        << "Output: ";
        ut::printArray(output[0], 2);
        // std::cout << "\ndOutput: ";
        // ut::printArray(dOutput[0], 2);

        const float cost = ml::cost::mean(output.data(), target.data(), 2, ml::cost::func::MeanSquaredError);

        std::cout << "\nCost: " << cost << '\n';
    }

    return 0;
}
