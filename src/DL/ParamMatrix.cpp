#include "DL/ParamMatrix.h"

using namespace ml;

ParamMatrix::ParamMatrix(float* data, size_t inputCount, size_t outputCount, ActFunc func) noexcept
    : inputCount(inputCount), outputCount(outputCount), data(data), rowCount(inputCount + 1), dataSize(rowCount * outputCount), func(func) {}

ParamMatrix& ParamMatrix::update(const Batch& input, const Batch& output, const Batch& dOutput, float learningRate)
{
    float* updatedWeights = new float[dataSize](); // Initialize to zero

    for (size_t i = 0; i < input.vectorCount(); i++)
    {
        const float* inputRow = input[i];
        const float* outputRow = output[i];
        const float* dOutputRow = dOutput[i];

        for (size_t j = 0; j < outputCount; j++)
        {
            const float dOutputNode = dOutputRow[j];
            if (dOutputNode == 0.0f) continue;

            const float* matrixRow = at(j);
            float* updatedWeightsRow = updatedWeights + (j * rowCount);

            size_t k = 0;
            for (; k < inputCount; k++)
            {
                updatedWeightsRow[k] += matrixRow[k] - learningRate * dOutputNode * inputRow[k];
            }
            updatedWeightsRow[k] += -learningRate * dOutputNode; // Bias
        }
    }

    for (size_t i = 0; i < dataSize; i++)
    {
        data[i] = updatedWeights[i] / static_cast<float>(input.vectorCount());
    }

    delete[] updatedWeights;

    return *this;
}

float* ParamMatrix::at(size_t row) { return data + (row * rowCount); }
const float* ParamMatrix::at(size_t row) const { return data + (row * rowCount); }

float* ParamMatrix::operator[](size_t row) noexcept { return at(row); }
const float* ParamMatrix::operator[](size_t row) const noexcept { return at(row); }