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

    /*
        @brief Prints an array in the format of a matrix.

        @param row The size of all rows.
        @param col The size of all collumns.
    */
    template<typename T>
    void printMatrix(const T* data, size_t row, size_t col)
    {
        std::cout << '\n';
        for (size_t i = 0; i < col; i++)
        {
            const T* dataRow = data + (i * row);

            for (size_t j = 0; j < row; j++)
            {
                std::cout << '|' << dataRow[j];
            }
            std::cout << "|\n";
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

