#pragma once

#include <memory>
#include <vector>

#include "DL/Cost.h"
#include "DL/Metaparams.h"
#include "DL/Propagation.h"

namespace ml
{
    class BasicModel;

    class BasicModel
    {
    private:
        // Holds weights and biases
        std::unique_ptr<float[]> _parameters = {};
        size_t _parametersSize = 0;
        Metaparams _metaparams = {};

    public:
        BasicModel();
        BasicModel(const Metaparams& metaparams);

        std::vector<float> forward(const std::vector<float>& inputs);

        // Gives an array that directly holds the model's parameters(weights and biases)
        float* parametersData() const;
        // Gives the total amount of all parameters(weights and biases)
        size_t parametersSize() const noexcept;
        // Gives a copy of the stored Metaparams
        Metaparams metaparameters() const;
    };

} // namespace ml
