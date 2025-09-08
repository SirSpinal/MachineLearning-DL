#include "DL/ParamMatrix.h"

using namespace ml;

ParamMatrix::ParamMatrix(float* data, size_t inputCount, size_t outputCount, ActFunc func) noexcept { set(data, inputCount, outputCount, func); }

ParamMatrix& ParamMatrix::update(const Batch& input, const Batch& output, const Batch& dOutput, float learningRate)
{
    actFuncPtr dActFunc = activation::getDerivActFunc(func);

    // Derivative of matrix parameters(this->data)
    float* gradients = new float[dataSize](); // Initialize to zero

    // Iterate through vectors in batches
    for (size_t i = 0; i < input.vectorCount(); i++)
    {
        const float* inputRow = input[i];
        const float* outputRow = output[i];
        const float* dOutputRow = dOutput[i];

        // Iterate through rows in matrix
        for (size_t j = 0; j < outputCount; j++)
        {
            const float dOutputNode = dOutputRow[j];
            if (dOutputNode == 0.0f) continue;

            const float* matrixRow = at(j);
            float* gradientRow = gradients + (j * rowCount);
            const float dFunc = dActFunc(outputRow[j]);

            const float commonDerivative = dOutputNode * dFunc;

            // Iterate through parameters in rows
            size_t k = 0;
            for (; k < inputCount; k++)
            {
                // total_weight_gradient += weight  - learning_rate * d_output * d_function * activation
                gradientRow[k] += matrixRow[k] - learningRate * commonDerivative * inputRow[k];
            }
            // total_bias_gradient += bias  - learning_rate * d_output * d_function
            gradientRow[k] += -learningRate * commonDerivative; // Bias
        }
    }

    // Sets new parameters to the mean of gradients.
    for (size_t i = 0; i < dataSize; i++)
    {
        // Division because we use the average gradient, not the total.
        data[i] = gradients[i] / static_cast<float>(input.vectorCount());
    }

    delete[] gradients;

    return *this;
}

ParamMatrix& ParamMatrix::set(float* data, size_t inputCount, size_t outputCount, ActFunc func) noexcept
{
    this->data;
    this->inputCount = inputCount;
    this->outputCount = outputCount;
    this->rowCount = inputCount + 1;
    this->dataSize = rowCount * outputCount;
    this->func = func;

    return *this;
}

float* ParamMatrix::at(size_t row) { return data + (row * rowCount); }
const float* ParamMatrix::at(size_t row) const { return data + (row * rowCount); }

float* ParamMatrix::operator[](size_t row) noexcept { return at(row); }
const float* ParamMatrix::operator[](size_t row) const noexcept { return at(row); }