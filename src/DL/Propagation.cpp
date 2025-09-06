#include "DL/Propagation.h"

using namespace ml;
using namespace ml::propagation;

fVector  ml::propagation::forward(const ParamMatrix& matrix, const fVector& input)
{
    actFuncPtr actFunc = activation::getActFunc(matrix.func);

    fVector output(matrix.outputCount, 0.0f);

    for (size_t i = 0; i < matrix.outputCount; i++)
    {
        const float* matrixRow = matrix[i];
        float& outputNode = output[i];
        
        size_t j = 0;
        for (; j < matrix.inputCount; j++)
        {
            const float weight = matrixRow[j];
            const float inputNode = input[j];
            outputNode += weight * inputNode;
        }
        outputNode = actFunc(outputNode + matrixRow[j]); // Bias and activation
    }

    return output;
}

Batch ml::propagation::forward(const ParamMatrix& matrix, const Batch& input)
{
    actFuncPtr actFunc = activation::getActFunc(matrix.func);
    Batch output(input.vectorCount(), matrix.outputCount);

    for (size_t i = 0; i < input.vectorCount(); i++)
    {
        const float* inputRow = input[i];
        float* outputRow = output[i];

        for (size_t j = 0; j < matrix.outputCount; j++)
        {
            const float* matrixRow = matrix[j];
            float& outputNode = outputRow[j];
        
            size_t k = 0;
            for (; k < matrix.inputCount; k++)
            {
                const float weight = matrixRow[k];
                const float inputNode = inputRow[k];
                outputNode += weight * inputNode;
            }
            outputNode = actFunc(outputNode + matrixRow[k]); // Bias and activation
        }
    }

    return output;
}

Batch ml::propagation::backward(const ParamMatrix& matrix, const Batch& output, const Batch& dOutput)
{
    actFuncPtr derivActFunc = activation::getDerivActFunc(matrix.func);

    Batch dInput(dOutput.vectorCount(), matrix.inputCount);

    for (size_t i = 0; i < dOutput.vectorCount(); i++)
    {
        const float* dOutputRow = dOutput[i];
        const float* outputRow = output[i];
        float* dInputRow = dInput[i];

        for (size_t j = 0; j < matrix.outputCount; j++)
        {
            const float* matrixRow = matrix[j];
            const float dOutputNode = dOutputRow[j];
            const float outputNode = outputRow[j];
            const float dFunc = derivActFunc(outputNode);

            for (size_t k = 0; k < matrix.inputCount; k++)
            {
                const float weight = matrixRow[k];
                dInputRow[k] += weight * dOutputNode * dFunc;
            }
        }
    }

    return dInput;
}
