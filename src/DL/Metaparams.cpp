#include "DL/Metaparams.h"

#include <cassert>

using namespace ml;

Metaparams::Metaparams() {}

Metaparams::Metaparams(std::initializer_list<size_t> layers, std::initializer_list<ActFunc> func)
    : _layerCount(layers.size()), _matrixCount(func.size()), _layers(std::make_unique<size_t[]>(layers.size())), _actFuncs(std::make_unique<ActFunc[]>(func.size()))
{
    assert((_matrixCount + 1) == _layerCount);

    size_t i = 0;
    for (; i < _matrixCount; i++)
    {
        _layers[i] = layers.begin()[i];
        _actFuncs[i] = func.begin()[i];
    }
    _layers[i] = layers.begin()[i];
}

size_t Metaparams::layerCount() const noexcept { return _layerCount; }
size_t Metaparams::matrixCount() const noexcept { return _matrixCount; }

size_t Metaparams::layers(size_t layerIndex) const noexcept
{
    assert(layerIndex < _layerCount);
    return _layers[layerIndex];
}
ActFunc Metaparams::act(size_t matrixIndex) const noexcept
{
    assert(matrixIndex < _matrixCount);
    return _actFuncs[matrixIndex];
}

