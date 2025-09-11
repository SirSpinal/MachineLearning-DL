#include "DL/Model.h"

#include <iostream>

#include "DL/InitFunc/ActInit.h"
#include "DL/Propagation.h"

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
                param = initFunction(inpCount, outCount);
            }
        }

        matrixIndex += matrixSize;
    }
}

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

float* BasicModel::parametersData() const { return _parameters.get(); }
size_t BasicModel::parametersSize() const noexcept { return _parametersSize; }
Metaparams BasicModel::metaparameters() const { return _metaparams; }

