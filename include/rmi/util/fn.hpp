#pragma once

#include <limits>
#include <type_traits>


/*======================================================================================================================
 * Bit Functions
 *====================================================================================================================*/

/** Computes the amount of bits needed to represent a given unsigned value. */
template<typename Numeric>
uint8_t bit_width(Numeric n)
{
    static_assert(std::is_unsigned<Numeric>::value, "not defined for signed integral types");

    /* Count leading zeros. */
    int lz;
    if constexpr (std::is_same_v<Numeric, unsigned>) {
        lz = __builtin_clz(n);
    } else if constexpr (std::is_same_v<Numeric, unsigned long>) {
        lz = __builtin_clzl(n);
    } else if constexpr (std::is_same_v<Numeric, unsigned long long>) {
        lz = __builtin_clzll(n);
    } else {
        static_assert(sizeof(Numeric) > sizeof(unsigned long long), "unsupported width of integral type");
    }

    return sizeof(Numeric) * 8 - lz;
}

/** Computes the length of the common prefix of two numeric values. */
template<typename Numeric>
uint8_t common_prefix_width(Numeric v1, Numeric v2)
{
    uint8_t length = std::numeric_limits<Numeric>::digits;

    while (v1 != v2) {
        v1 >>= 1;
        v2 >>= 1;
        length -= 1;
    }
    return length;
}

