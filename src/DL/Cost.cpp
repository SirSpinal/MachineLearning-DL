#include "DL/Cost.h"

#include <cmath>

using namespace ml;

#pragma region Namespace functions
float ml::cost::total(const float* predicted, const float* target, size_t count, costFuncPtr costFunc)
{
    float totalCost = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        totalCost += costFunc(predicted[i], target[i]);
    }
    return totalCost;
}

float ml::cost::mean(const float* predicted, const float* target, size_t count, costFuncPtr costFunc)
{
    return total(predicted, target, count, costFunc) / static_cast<float>(count);
}

void ml::cost::apply(float* array, const float* predicted, const float* target, size_t count, costFuncPtr costFunc)
{
    for (size_t i = 0; i < count; i++)
    {
        array[i] = costFunc(predicted[i], target[i]);
    }
}
#pragma endregion

#pragma region Cost functions
namespace ml
{
namespace cost
{
    namespace func
    {
        float MeanSquaredError(float predicted, float target) { return std::pow(predicted - target, 2.0f); }
        float BinaryCrossEntropy(float predicted, float target)
        {
            return -(target * std::log(predicted) + (1 - target) * std::log(1 - predicted));
        }
    
    } // namespace func
    
    namespace deriv
    {
        float MeanSquaredError(float predicted, float target) { return 2.0f * (predicted - target); }
        float BinaryCrossEntropy(float predicted, float target) { return predicted - target; }
    
    } // namespace deriv

} // namespace cost

} // namespace ml
#pragma endregion
