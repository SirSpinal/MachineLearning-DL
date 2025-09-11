#pragma once

#include <iostream>

namespace ut
{
    template<typename T>
    void printArray(const T* array, size_t count)
    {
        std::cout << '|';
        for (size_t i = 0; i < count; i++)
        {
            std::cout << array[i] << '|';
        }
    }

    template<typename T>
    void printParamMatrix(const T* data, size_t inp, size_t out)
    {
        std::cout << '\n';
        for (size_t i = 0; i < out; i++)
        {
            const T* dataRow = data + (i * (inp + 1));

            size_t j = 0;
            for (; j < inp; j++)
            {
                std::cout << dataRow[j] << '|';
            }
            std::cout << ';' << dataRow[j] << '\n';
        }
    }

} // namespace ut

