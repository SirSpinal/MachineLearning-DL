#pragma once

#include <memory>
#include <vector>

#include "DL/Cost.h"
#include "DL/Batch.h"
#include "DL/Metaparams.h"

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

        /*
            @brief Gives all the outputs of a forward prop from every layer.

            @param inputs The inputs used in the forward propagation.

            @return A vector containing the outputs. In the order of first layer to last layer.
        */
        std::vector<Batch> _forwardpropagate(const Batch& inputs) const;
        
        public:
        BasicModel();
        BasicModel(const Metaparams& metaparams);
        
        /*
            @brief Does a full forward propagation with the inputs. Uses the model's parameters(weights/biases) for the forward propagation.

            @param inputs The inputs used in the forward propagation.

            @return The output of the forward propagation.
        */
        std::vector<float> forward(const std::vector<float>& inputs) const;

        BasicModel& update(const Batch& inputs, const Batch& targets, costFuncPtr dCostFunc, float learningRate=0.01f);

        // Gives an array that directly holds the model's parameters(weights and biases)
        float* parametersData() const;
        // Gives the total amount of all parameters(weights and biases)
        size_t parametersSize() const noexcept;
        // Gives a copy of the stored Metaparams
        Metaparams metaparameters() const;
    };

} // namespace ml
