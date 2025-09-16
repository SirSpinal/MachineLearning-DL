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
Metaparams::Metaparams(const Metaparams& other) { copy(other); }
Metaparams::Metaparams(Metaparams&& other) noexcept { move(std::move(other)); }

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

Metaparams& Metaparams::copy(const Metaparams& other)
{
    if (other._layerCount != _layerCount)
    {
        _layerCount = other._layerCount;
        _matrixCount = other._matrixCount;

        _layers.reset(new size_t[_layerCount]);
        _actFuncs.reset(new ActFunc[_matrixCount]);
    }

    size_t i = 0;
    for (; i < _matrixCount; i++)
    {
        _layers[i] = other._layers[i];
        _actFuncs[i] = other._actFuncs[i];
    }
    _layers[i] = other._layers[i];

    return *this;
}
Metaparams& Metaparams::move(Metaparams&& other) noexcept
{
    _layerCount = other._layerCount;
    _matrixCount = other._matrixCount;
    _layers = std::move(other._layers);
    _actFuncs = std::move(other._actFuncs);

    other._layerCount = 0;
    other._matrixCount = 0;

    return *this;
}

Metaparams& Metaparams::operator=(const Metaparams& other) { return copy(other); }
Metaparams& Metaparams::operator=(Metaparams&& other) noexcept { return copy(std::move(other)); }
