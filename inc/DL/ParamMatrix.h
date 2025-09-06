#pragma once

#include <cstddef>

#include "DL/ActFunc/ActFunc.h"
#include "DL/Batch.h"

namespace ml
{
    struct ParamMatrix
    {
        /*
            A matrix, not owned by this class.
            In the format [row1, row2, ..., rowN]
            where each row is [w1, w2, ..., wM, b] (weights + bias)
        */
        float* data = nullptr;
        size_t inputCount = 0;
        size_t outputCount = 0;
        size_t rowCount = 0;
        size_t dataSize = 0;
        ActFunc func = ActFunc::None;

        ParamMatrix(float* data, size_t inputCount, size_t outputCount, ActFunc func) noexcept;

        ParamMatrix& update(const Batch& input, const Batch& output, const Batch& dOutput, float learningRate=0.01f);

        float* at(size_t row);
        const float* at(size_t row) const;

        float* operator[](size_t row) noexcept;
        const float* operator[](size_t row) const noexcept;
    };

} // namespace ml

