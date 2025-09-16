#pragma once

#include <cstddef>

namespace ml
{
    using costFuncPtr = float(*)(float predicted, float target);

    namespace cost
    {
        float total(const float* predicted, const float* target, size_t count, costFuncPtr costFunc);
        float mean(const float* predicted, const float* target, size_t count, costFuncPtr costFunc);
        /*
            @brief Applies the cost function to each element in the `array` parameter.

            @param array The array to store the result.
            @param predicted The predicted values.
            @param target The target values.
            @param count The number of elements in all of the arrays.
            @param costFunc The cost function to apply on the elements.
        */
        void apply(float* array, const float* predicted, const float* target, size_t count, costFuncPtr costFunc);

        namespace func
        {
            float MeanSquaredError(float predicted, float target);
            float BinaryCrossEntropy(float predicted, float target);

        } // namespace func

        namespace deriv
        {
            float MeanSquaredError(float predicted, float target);
            float BinaryCrossEntropy(float predicted, float target);

        } // namespace deriv

    } // namespace cost

} // namespace ml

