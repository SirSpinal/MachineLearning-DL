#pragma once

#include <cmath>

namespace ml
{
namespace activation
{
    namespace functions
    {
        float None(float x) { return x; }
        float ReLU(float x) { return x > 0.0f ? x : 0.0f; }
        float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
        float Tanh(float x) { return std::tanh(x); }

    } // namespace functions
    
    namespace derivatives
    {
        float None([[maybe_unused]] float) { return 1.0f; }
        float ReLU(float x) { return x > 0.0f ? 1.0f : 0.0f; }
        float Sigmoid(float x) 
        { 
            float sig = 1.0f / (1.0f + std::exp(-x));
            return sig * (1.0f - sig); 
        }
        float Tanh(float x) 
        { 
            float t = std::tanh(x);
            return 1.0f - t * t; 
        }

    } // namespace derivatives

} // namespace activation

} // namespace ml

