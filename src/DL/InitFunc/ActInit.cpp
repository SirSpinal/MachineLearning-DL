#include "DL/InitFunc/ActInit.h"

using namespace ml;

initFuncPtr ml::init::getInitFunc(ActFunc func)
{
    using e = ml::ActFunc;

    switch (func)
    {
    case e::ReLU:
        return init::func::Kaiming_Normal;

    default:
        return init::func::Xavier_Normal;
    }
}
