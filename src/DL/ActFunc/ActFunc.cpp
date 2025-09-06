#include "DL/ActFunc/ActFunc.h"

#include "DL/ActFunc/Functions.h"

using namespace ml;

actFuncPtr ml::activation::getActFunc(ActFunc af)
{
    switch (af)
    {
    case ActFunc::None: return functions::None;
    case ActFunc::ReLU: return functions::ReLU;
    case ActFunc::Sigmoid: return functions::Sigmoid;
    case ActFunc::Tanh: return functions::Tanh;
    default: return functions::None;
    }
    
}

actFuncPtr ml::activation::getDerivActFunc(ActFunc af)
{
    switch (af)
    {
    case ActFunc::None: return derivatives::None;
    case ActFunc::ReLU: return derivatives::ReLU;
    case ActFunc::Sigmoid: return derivatives::Sigmoid;
    case ActFunc::Tanh: return derivatives::Tanh;
    default: return derivatives::None;
    }

}
