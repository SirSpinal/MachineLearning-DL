#pragma once

#include "DL/InitFunc/Functions.h"
#include "DL/ActFunc/ActFunc.h"

namespace ml
{
    namespace init
    {
        /*
            @brief Gives the recommended init function according to the activation function.

            @param ActFunc The activation function to find the recommended initiation function.

            @return The recommended initiation function.
        */
        initFuncPtr getInitFunc(ActFunc func);

    } // namespace init

} // namespace ml
