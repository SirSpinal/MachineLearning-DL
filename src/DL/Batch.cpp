#include "DL/Batch.h"

#include <cassert>

using namespace ml;

#pragma region Constructors
Batch::Batch() {}
ml::Batch::Batch(size_t count, size_t size, float fillValue)
: _data(std::make_unique<float[]>(count * size)), _count(count), _size(size)
{
    for (size_t i = 0; i < _size; i++)
    {
        _data[i] = fillValue;
    }
}

ml::Batch::Batch(float* data, size_t count, size_t size) : _data(data), _count(count), _size(size) {}

ml::Batch::Batch(std::unique_ptr<float[]>& data, size_t count, size_t size) : _count(count), _size(size)
{
    _data = std::move(data);
}

ml::Batch::Batch(std::initializer_list<std::initializer_list<float>> matrix)
: _data(std::make_unique<float[]>(matrix.size() * matrix.begin()->size())), _count(matrix.size()), _size(matrix.begin()->size())
{
    for (size_t i = 0; i < _count; i++)
    {
        const float* initList = matrix.begin()[i].begin();
        float* start = at(i);

        for (size_t j = 0; j < _size; j++)
        {
            start[j] = initList[j];
        }
    }
}

ml::Batch::Batch(const Batch& other) { copy(other); }
ml::Batch::Batch(Batch&& other) noexcept { move(std::move(other)); }
#pragma endregion

#pragma region Methods
float* ml::Batch::at(size_t vectorIndex) const noexcept
{
    assert(vectorIndex < _count);
    return _data.get() + (vectorIndex * _size);
}

fVector ml::Batch::get(size_t vectorIndex) const
{
    const size_t rowStart = vectorIndex * _size;
    fVector dataVector(_size);

    for (size_t i = 0; i < _size; i++)
    {
        dataVector[i] = _data[rowStart + i];
    }

    return dataVector;
}

Batch& ml::Batch::set(size_t vectorIndex, const fVector& dataVector)
{
    const size_t rowStart = vectorIndex * _size;

    for (size_t i = 0; i < _size; i++)
    {
        _data[rowStart + i] = dataVector[i];
    }

    return *this;
}

size_t ml::Batch::vectorCount() const noexcept
{
    return _count;
}

size_t ml::Batch::vectorSize() const noexcept
{
    return _size;
}

float* ml::Batch::data() const noexcept
{
    return _data.get();
}

Batch& ml::Batch::copy(const Batch& other)
{
    _count = other._count;
    _size = other._size;

    const size_t dataSize = _count * _size;

    _data.reset(new float[dataSize]);

    for (size_t i = 0; i < dataSize; i++)
    {
        _data[i] = other._data[i];
    }

    return *this;
}

Batch& ml::Batch::move(Batch&& other) noexcept
{
    _data.reset();

    _data = std::move(other._data);
    _count = other._count;
    _size = other._size;

    other._count = 0;
    other._size = 0;

    return *this;
}
#pragma endregion

#pragma region Operators
Batch& ml::Batch::operator=(const Batch& other)
{
    copy(other);
    return *this;
}

Batch& ml::Batch::operator=(Batch&& other)
{
    move(std::move(other));
    return *this;
}

float* ml::Batch::operator[](size_t vectorIndex) const noexcept { return at(vectorIndex); }
#pragma endregion
