#include "DL/InitFunc/Functions.h"

#include <cmath>
#include <random>

#include "Settings/Settings.h"

using namespace ml;

std::mt19937& getRng()
{
    static const unsigned int SEED = settings::USE_CUSTOM_RNG_SEED ? settings::RNG_SEED : std::random_device{}();
    static std::mt19937 generator(SEED);
    return generator;
};

namespace ml
{
namespace init
{
    namespace func
    {
        float Kaiming_Normal(size_t inputCount, [[maybe_unused]] size_t)
        {
            const float x = std::sqrt(2.0f / static_cast<float>(inputCount));
            std::normal_distribution<float> dist(0.0f, x);
            return dist(getRng());
        }
        float Kaiming_Uniform(size_t inputCount, [[maybe_unused]] size_t)
        {
            const float x = std::sqrt(2.0f / static_cast<float>(inputCount));
            std::uniform_real_distribution<float> dist(-x, x);
            return dist(getRng());
        }

        float Xavier_Normal(size_t inputCount, size_t outputCount)
        {
            const float x = 6.0f / (inputCount + outputCount);
            std::normal_distribution<float> dist(0.0f, x);
            return dist(getRng());
        }
        float Xavier_Uniform(size_t inputCount, size_t outputCount)
        {
            const float x = 6.0f / (inputCount + outputCount);
            std::uniform_real_distribution<float> dist(-x, x);
            return dist(getRng());
        }

    } // namespace func
    
} // namespace init
    
} // namespace ml;

