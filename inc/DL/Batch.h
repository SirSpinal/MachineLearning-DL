#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

namespace ml
{
    using fVector = std::vector<float>;

    class Batch
    {
    private:
        std::unique_ptr<float[]> _data = nullptr;
        size_t _count = 0; // Amount of data vectors
        size_t _size = 0; // Size of data vectors

    public:
        Batch(size_t count, size_t size, float fillValue=0.0f);
        Batch(float* data, size_t count, size_t size);
        Batch(std::unique_ptr<float[]>& data, size_t count, size_t size);
        Batch(std::initializer_list<std::initializer_list<float>> matrix);
        Batch(const Batch& other);
        Batch(Batch&& other) noexcept;
        
        float* at(size_t vectorIndex) const noexcept;
        
        fVector get(size_t vectorIndex) const;
        Batch& set(size_t vectorIndex, const fVector& dataVector);
        
        size_t vectorCount() const noexcept;
        size_t vectorSize() const noexcept;
        
        float* data() const noexcept;

        Batch& copy(const Batch& other);
        Batch& move(Batch&& other) noexcept;

        Batch& operator=(const Batch& other);
        Batch& operator=(Batch&& other);

        float* operator[](size_t vectorIndex) const noexcept;
    };

} // namespace ml

