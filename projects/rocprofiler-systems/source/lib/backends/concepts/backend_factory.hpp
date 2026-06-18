#pragma once

#include <concepts>

namespace rocprofsys::backends::concepts
{

template <typename T>
concept backend_factory = requires {
    typename T::backend_t;
    T::create();
};

}  // namespace rocprofsys::backends::concepts
