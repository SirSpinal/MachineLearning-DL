#pragma once

namespace ml
{
    using actFuncPtr = float(*)(float);

    enum class ActFunc
    {
        None,
        ReLU,
        Sigmoid,
        Tanh
    };

    namespace activation
    {
        actFuncPtr getActFunc(ActFunc af);
        actFuncPtr getDerivActFunc(ActFunc af);

    } // namespace activation

} // namespace ml

