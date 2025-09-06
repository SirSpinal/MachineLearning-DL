#pragma once

#include "DL/Batch.h"
#include "DL/ParamMatrix.h"

namespace ml
{
    namespace propagation
    {
        using fVector = std::vector<float>;

        fVector forward(const ParamMatrix& matrix, const fVector& input);
        Batch forward(const ParamMatrix& matrix, const Batch& input);
        Batch backward(const ParamMatrix& matrix, const Batch& output, const Batch& dOutput);

    } // namespace propagation

} // namespace ml
