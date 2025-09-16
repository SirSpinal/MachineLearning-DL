#pragma once

#include "DL/Batch.h"
#include "DL/ParamMatrix.h"

namespace ml
{
    namespace propagation
    {
        std::vector<float> forward(const ParamMatrix& matrix, const std::vector<float>& input);
        Batch forward(const ParamMatrix& matrix, const Batch& input);
        Batch backward(const ParamMatrix& matrix, const Batch& output, const Batch& dOutput);

    } // namespace propagation

} // namespace ml
