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
        std::unique_ptr<float[]> _data = nullptr; // A matrix
        size_t _count = 0; // Amount of data vectors
        size_t _size = 0; // Size of data vectors

    public:
        Batch(size_t count, size_t size, float fillValue=0.0f);
        Batch(float* data, size_t count, size_t size);
        Batch(std::unique_ptr<float[]>& data, size_t count, size_t size);
        Batch(std::initializer_list<std::initializer_list<float>> matrix);
        Batch(const Batch& other);
        Batch(Batch&& other) noexcept;

        // Returns an array pointing to the specified vector.
        float* at(size_t vectorIndex) const noexcept;
        
        // Returns a copy of the specified vector.
        fVector get(size_t vectorIndex) const;
        // Sets the specified vector in the batch to the provided data vector.
        Batch& set(size_t vectorIndex, const fVector& dataVector);

        // Returns the number of vectors in the batch.
        size_t vectorCount() const noexcept;
        // Returns the size of each vector in the batch.
        size_t vectorSize() const noexcept;
        
        float* data() const noexcept;

        Batch& copy(const Batch& other);
        Batch& move(Batch&& other) noexcept;

        Batch& operator=(const Batch& other);
        Batch& operator=(Batch&& other);

        float* operator[](size_t vectorIndex) const noexcept;
    };

} // namespace ml

