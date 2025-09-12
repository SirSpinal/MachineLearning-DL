#include <iostream>

#include "DL/Model.h"

#include "util/Print.h"
#include "util/PrintModel.h"

int main()
{
    ml::BasicModel x(
        ml::Metaparams(
            {2, 3, 1},
            {ml::ActFunc::ReLU, ml::ActFunc::None}
        )
    );

    ut::printBasicModelData(x);

    ml::Batch input = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
    };

    // auto result = x._forwardpropagate(input);

    // for (size_t i = 0; i < result.size(); i++)
    // {
    //     ml::Batch& b = result[i];
    //     std::cout << "\nResult [" << i << "]: " << b.vectorSize();
    //     ut::printMatrix(b.data(), b.vectorSize(), b.vectorCount());
    // }

    return 0;
}

/*
// Training a model of only 2 layers: input layer (2 nodes) and output layer (2 nodes)

    std::system("cls");

    constexpr size_t EPOCH = 128;

    ml::fVector parameters = {
        // Matrix 1
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,

        // Matrix 2,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,
        ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), ml::init::func::Kaiming_Normal(2, 0),ml::init::func::Kaiming_Normal(2, 0), 0.0f,

        // Matrix 3
        ml::init::func::Xavier_Normal(3, 1), ml::init::func::Xavier_Normal(3, 1), ml::init::func::Xavier_Normal(3, 1), 0.0f,
    };

    ml::ParamMatrix matrix1(parameters.data(), 2, 4, ml::ActFunc::ReLU);
    ml::ParamMatrix matrix2(parameters.data() + 12, 4, 3, ml::ActFunc::ReLU);
    ml::ParamMatrix matrix3(parameters.data() + 12 + 15, 3, 1, ml::ActFunc::None);

    ml::Batch input = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
    };

    ml::Batch target = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f},
    };

    for (size_t i = 0; i < EPOCH; i++)
    {
        // Get outputs
        ml::Batch output1 = ml::propagation::forward(matrix1, input);
        ml::Batch output2 = ml::propagation::forward(matrix2, output1);
        ml::Batch output3 = ml::propagation::forward(matrix3, output2);
        
        // Get gradients
        ml::Batch dOutput3(target.vectorCount(), matrix3.outputCount);
        ml::cost::apply(dOutput3.data(), output3.data(), target.data(), target.vectorCount() * target.vectorSize(), ml::cost::deriv::MeanSquaredError);
        
        ml::Batch dOutput2 = ml::propagation::backward(matrix3, output3, dOutput3);
        ml::Batch dOutput1 = ml::propagation::backward(matrix2, output2, dOutput2);

        // Update
        matrix3.update(output2, output3, dOutput3);
        matrix2.update(output1, output2, dOutput2);
        matrix1.update(input, output1, dOutput1);

        // Print
        if ((i + 1) % 4 == 0) { continue; }

        std::cout
        << "\n[" << (i + 1) << "]:\n"
        << "Output: ";
        ut::printArray(output3[0], output3.vectorSize());

        const float cost = ml::cost::mean(output3.data(), target.data(), target.vectorCount() * target.vectorSize(), ml::cost::func::MeanSquaredError);

        std::cout << "\nCost: " << cost << '\n';
    }

    std::cout << "\nPARAMETERS:\n";
    ut::printParamMatrix(matrix1.data, matrix1.inputCount, matrix1.outputCount);
    ut::printParamMatrix(matrix2.data, matrix1.inputCount, matrix1.outputCount);
    ut::printParamMatrix(matrix2.data, matrix1.inputCount, matrix1.outputCount);

    ml::Batch output1 = ml::propagation::forward(matrix1, input);
    ml::Batch output2 = ml::propagation::forward(matrix2, output1);
    ml::Batch output3 = ml::propagation::forward(matrix3, output2);
    std::cout << "\nOUTPUT:\n";
    ut::printArray(target.data(), target.vectorCount() * target.vectorSize());
*/
