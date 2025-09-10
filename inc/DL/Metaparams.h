#pragma once

#include <initializer_list>
#include <memory>

#include "DL/ActFunc/ActFunc.h"

namespace ml
{
    class Metaparams;

    /*
        The `Metaparams` class is used to hold important information in model classes,
        such as: layer count, specific layer sizes, matrix count, and activation functions.
    */
    class Metaparams
    {
    private:
        size_t _layerCount = 0;
        size_t _matrixCount = 0;
        std::unique_ptr<size_t[]> _layers = {};
        std::unique_ptr<ActFunc[]> _actFuncs = {};

    public:
        Metaparams();
        // `func.size()` must be the same as `layers.size() - 1`.
        Metaparams(std::initializer_list<size_t> layers, std::initializer_list<ActFunc> func);

        size_t layerCount() const noexcept;
        size_t matrixCount() const noexcept;

        // `layerIndex` must b e under layer count.
        size_t layers(size_t layerIndex) const noexcept;
        // `matrixIndex` must b e under matrix count.
        ActFunc act(size_t matrixIndex) const noexcept;
    };

} // namespace ml

