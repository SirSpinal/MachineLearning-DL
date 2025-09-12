#include "DL/Model.h"

#include <cassert>
#include <iostream>

#include "DL/InitFunc/ActInit.h"
#include "DL/Propagation.h"

#include "util/Print.h"

using namespace ml;

BasicModel::BasicModel() {}
BasicModel::BasicModel(const Metaparams& metaparams) : _metaparams(metaparams)
{
    for (size_t i = 0; i < _metaparams.matrixCount(); i++)
    {
        _parametersSize += (_metaparams.layers(i) + 1) * _metaparams.layers(i + 1);
    }

    _parameters = std::make_unique<float[]>(_parametersSize);

    for (size_t i = 0, matrixIndex = 0; i < _metaparams.matrixCount(); i++)
    {
        const size_t inpCount = _metaparams.layers(i);
        const size_t outCount = _metaparams.layers(i + 1);
        const size_t rowSize = inpCount + 1;

        const size_t matrixSize = rowSize * outCount;
        const ActFunc actFunc = _metaparams.act(i);
        initFuncPtr initFunction = init::getInitFunc(actFunc);

        for (size_t j = 0; j < matrixSize; j++)
        {
            const bool isBias = (((j + 1) % rowSize) == 0);
            float& param = _parameters[matrixIndex + j];

            if (isBias)
            {
                param = 0.0f;
            }
            else
            {
                //param = 1.0f;
                param = initFunction(inpCount, outCount);
            }
        }

        matrixIndex += matrixSize;
    }
}

#pragma region Methods




#pragma region _private
std::vector<Batch> BasicModel::_forwardpropagate(const Batch& inputs) const
{
    std::vector<Batch> inputTensor(_metaparams.layerCount());
    Batch current = inputs;

    size_t i = 0;
    for (size_t matrixIndex = 0; i < _metaparams.matrixCount(); i++)
    {
        const size_t inpCount = _metaparams.layers(i);
        const size_t outCount = _metaparams.layers(i + 1);
        const size_t rowSize = inpCount + 1;

        const size_t matrixSize = rowSize * outCount;
        const ActFunc actFunc = _metaparams.act(i);
        float* matrixData = &(_parameters[matrixIndex]);

        const ParamMatrix matrix(matrixData, inpCount, outCount, actFunc);

        inputTensor[i] = current;
        current = propagation::forward(matrix, current);

        matrixIndex += matrixSize;
    }
    inputTensor[i] = current;

    return inputTensor;
}
#pragma endregion

std::vector<float> BasicModel::forward(const std::vector<float>& input) const
{
    std::vector<float> current = input;

    for (size_t i = 0, matrixIndex = 0; i < _metaparams.matrixCount(); i++)
    {
        const size_t inpCount = _metaparams.layers(i);
        const size_t outCount = _metaparams.layers(i + 1);
        const size_t rowSize = inpCount + 1;

        const size_t matrixSize = rowSize * outCount;
        const ActFunc actFunc = _metaparams.act(i);
        float* matrixData = &(_parameters[matrixIndex]);

        const ParamMatrix matrix(matrixData, inpCount, outCount, actFunc);

        current = propagation::forward(matrix, current);

        matrixIndex += matrixSize;
    }

    return current;
}

BasicModel& BasicModel::update(const Batch& inputs, const Batch& targets, costFuncPtr dCostFunc, float learningRate)
{
    assert(inputs.vectorSize() == _metaparams.layers(0));
    assert(targets.vectorSize() == _metaparams.layers(_metaparams.matrixCount()));
    assert(inputs.vectorCount() == targets.vectorCount());

    std::vector<Batch> inputData = _forwardpropagate(inputs);
    Batch currentGradient(targets.vectorCount(), targets.vectorSize());
    {
        const Batch& output = inputData.back();

        ml::cost::apply(
            currentGradient.data(), // To change
            output.data(), // Model predictions
            targets.data(), // Target values
            targets.vectorCount() * targets.vectorSize(), // Data size
            dCostFunc // Cost function derivative
        );
    }

    for (size_t i = _metaparams.matrixCount() - 1, matrixIndex = _parametersSize; true; i--)
    {
        const size_t inpCount = _metaparams.layers(i);
        const size_t outCount = _metaparams.layers(i + 1);
        const size_t rowSize = inpCount + 1;

        const size_t matrixSize = rowSize * outCount;
        const ActFunc actFunc = _metaparams.act(i);
        
        // Change matrix index
        matrixIndex -= matrixSize;

        // Param matrix
        float* matrixData = &(_parameters[matrixIndex]);
        ParamMatrix matrix(matrixData, inpCount, outCount, actFunc);

        // Input and output for current matrix
        const Batch& matrixInput = inputData[i];
        const Batch& matrixOutput = inputData[i + 1];

        // Update
        matrix.update(matrixInput, matrixOutput, currentGradient, learningRate);

        if (i == 0) { break; }

        // Propagate backwards, if second layer from first.
        currentGradient = ml::propagation::backward(matrix, matrixOutput, currentGradient);
    }

    return *this;
}

float* BasicModel::parametersData() const { return _parameters.get(); }
size_t BasicModel::parametersSize() const noexcept { return _parametersSize; }
Metaparams BasicModel::metaparameters() const { return _metaparams; }
#pragma endregion
