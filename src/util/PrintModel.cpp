#include "util/PrintModel.h"

#include <iostream>

#include "util/Print.h"

using namespace ut;

void ut::printBasicModelData(ml::BasicModel& model)
{
    ml::Metaparams metaparams = model.metaparameters();
    const float* params = model.parametersData();

    std::cout << "\nLayer count: " << metaparams.layerCount();
    std::cout << "\nMatrix count: "<< metaparams.matrixCount();
    std::cout << "\nLayers: ";
    for (size_t i = 0; i < metaparams.layerCount(); i++)
    {
        std::cout << metaparams.layers(i) << '|';
    }
    std::cout << "\nAct functions: ";
    for (size_t i = 0; i < metaparams.matrixCount(); i++)
    {
        std::cout << (int64_t)metaparams.act(i) << '|';
    }
    std::cout << "\nMatrix parameters:\n\n";

    for (size_t i = 0, matrixIndex = 0; i < metaparams.matrixCount(); i++)
    {
        const size_t inpCount = metaparams.layers(i);
        const size_t outCount = metaparams.layers(i + 1);

        std::cout << "Matrix [" << (i + 1) << "]:";
        printParamMatrix<float>(params + matrixIndex, inpCount, outCount);
        std::cout << "\n";

        matrixIndex += (inpCount + 1) * outCount;
    }
}
