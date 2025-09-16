#pragma once

#include <cstddef>

namespace ml
{
    // To use an init function, use the output of an init function as the weight/bias's value.
    using initFuncPtr = float(*)(size_t inputCount, size_t outputCount);

    namespace init
    {
        namespace func
        {
            float Kaiming_Normal(size_t inputCount, [[maybe_unused]] size_t);
            float Kaiming_Uniform(size_t inputCount, [[maybe_unused]] size_t);
    
            float Xavier_Normal(size_t inputCount, size_t outputCount);
            float Xavier_Uniform(size_t inputCount, size_t outputCount);            
        } // namespace func        

    } // namespace init_func

} // namespace ml

