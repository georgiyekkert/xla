/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_H_

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__GNUC__) || defined(__clang__)) && defined(__SSE2__)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && defined(_M_X64)
#define XLA_HAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define XLA_HAS_SSE2
#elif defined(__AVX__)
#define XLA_HAS_SSE2
#endif

#ifdef XLA_HAS_SSE2
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

namespace xla {

// Generic transpose kernel.
//
// All of the kernels that follow in this file are optimized versions of this
// generic kernel, specialized to particular block sizes and data types.
//
// The transpose kernel requires its input to be contiguous in one of the two
// dimensions being transposed, and the output to be contiguous in the other
// dimension.
//
// lda, ldb are strides in bytes.
template <typename T, int bs>
struct TransposeMicroKernel {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    for (int i = 0; i < bs; ++i) {
      for (int j = 0; j < bs; ++j) {
        *reinterpret_cast<T*>(b + i * ldb + j * sizeof(T)) =
            *reinterpret_cast<T const*>(a + j * lda + i * sizeof(T));
      }
    }
  }
};

#pragma push_macro("XLA_UNROLL")
#if defined(__clang__)
#define XLA_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define XLA_UNROLL _Pragma("GCC unroll 128")
#else
#define XLA_UNROLL
#endif

#pragma push_macro("XLA_ALWAYS_INLINE")
#if defined(__GNUC__) || defined(__clang__)
#define XLA_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define XLA_ALWAYS_INLINE __forceinline
#else
#define XLA_ALWAYS_INLINE
#endif

enum class Extract { kLo, kHi };

#ifdef __AVX__
template <size_t element_size, Extract>
__m256i Unpack(__m256i a, __m256i b);

#if defined(__AVX2__)
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<1, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi8(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<1, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi8(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<2, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi16(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<2, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi16(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<4, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi32(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<4, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi32(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<8, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_unpacklo_epi64(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<8, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_unpackhi_epi64(a, b);
}
#else
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<1, Extract::kLo>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpacklo_epi8(a_hi, b_hi);
  __m128i lo = _mm_unpacklo_epi8(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<1, Extract::kHi>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpackhi_epi8(a_hi, b_hi);
  __m128i lo = _mm_unpackhi_epi8(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<2, Extract::kLo>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpacklo_epi16(a_hi, b_hi);
  __m128i lo = _mm_unpacklo_epi16(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<2, Extract::kHi>(__m256i a, __m256i b) {
  __m128i a_hi = _mm256_extractf128_si256(a, 1);
  __m128i b_hi = _mm256_extractf128_si256(b, 1);
  __m128i a_lo = _mm256_castsi256_si128(a);
  __m128i b_lo = _mm256_castsi256_si128(b);
  __m128i hi = _mm_unpackhi_epi16(a_hi, b_hi);
  __m128i lo = _mm_unpackhi_epi16(a_lo, b_lo);
  return _mm256_set_m128i(hi, lo);
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<4, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_castps_si256(
      _mm256_unpacklo_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<4, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_castps_si256(
      _mm256_unpackhi_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}

template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<8, Extract::kLo>(__m256i a, __m256i b) {
  return _mm256_castpd_si256(
      _mm256_unpacklo_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)));
}
template <>
XLA_ALWAYS_INLINE inline __m256i Unpack<8, Extract::kHi>(__m256i a, __m256i b) {
  return _mm256_castpd_si256(
      _mm256_unpackhi_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)));
}
#endif
#endif

#ifdef XLA_HAS_SSE2
template <size_t element_size, Extract>
__m128i Unpack(__m128i a, __m128i b);

template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<1, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi8(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<1, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi8(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<2, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi16(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<2, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi16(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<4, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi32(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<4, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi32(a, b);
}

template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<8, Extract::kLo>(__m128i a, __m128i b) {
  return _mm_unpacklo_epi64(a, b);
}
template <>
XLA_ALWAYS_INLINE inline __m128i Unpack<8, Extract::kHi>(__m128i a, __m128i b) {
  return _mm_unpackhi_epi64(a, b);
}

template <size_t element_size, std::size_t step_size, typename T, std::size_t N>
XLA_ALWAYS_INLINE inline std::array<T, N> UnpackStep(
    const std::array<T, N>& last_transpose) {
  static_assert(N % (step_size * 2) == 0);
  std::array<T, N> unpack;
  XLA_UNROLL
  for (int i = 0; i < N; i += step_size * 2) {
    XLA_UNROLL
    for (int j = 0; j < step_size; ++j) {
      unpack[i + 2 * j + 0] = Unpack<element_size * step_size, Extract::kLo>(
          last_transpose[i + j], last_transpose[i + j + step_size]);
      unpack[i + 2 * j + 1] = Unpack<element_size * step_size, Extract::kHi>(
          last_transpose[i + j], last_transpose[i + j + step_size]);
    }
  }
  return unpack;
}

template <size_t element_size, std::size_t step_size, size_t max_step_size,
          typename T, std::size_t N>
XLA_ALWAYS_INLINE inline std::array<T, N> UnpackSequence(
    const std::array<T, N>& last_transpose) {
  if constexpr (element_size * step_size <= max_step_size) {
    std::array<T, N> unpack =
        UnpackStep<element_size, step_size>(last_transpose);
    return UnpackSequence<element_size, step_size * 2, max_step_size>(unpack);
  }
  return last_transpose;
}

template <typename T, int bs>
struct Sse2SquareTransposeMicroKernelImpl {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size == sizeof(__m128i));
    std::array<__m128i, bs> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      last_transpose[i] =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + lda * i));
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/8>(
            last_transpose);

    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(b + ldb * i),
                       last_transpose[i]);
    }
  }
};
#endif

#ifdef __AVX__
template <typename T, int bs>
struct AvxSquareTransposeMicroKernelImpl {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size == sizeof(__m256i));
    std::array<__m256i, bs> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* row0_low = reinterpret_cast<const __m128i*>(a + lda * (i + 0));
      auto* row0_high = row0_low + 1;
      auto* row1_low = reinterpret_cast<const __m128i*>(a + lda * (i + bs / 2));
      auto* row1_high = row1_low + 1;
      last_transpose[i] = _mm256_loadu2_m128i(row1_low, row0_low);
      last_transpose[i + bs / 2] = _mm256_loadu2_m128i(row1_high, row0_high);
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/8>(
            last_transpose);

    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(b + ldb * i),
                          last_transpose[i]);
    }
  }
};
#endif

#ifdef __AVX__
template <typename T, int bs>
struct AvxRectangularTransposeMicroKernelImpl {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    constexpr size_t element_size = sizeof(T);
    static_assert(element_size <= 16);
    static_assert(16 % element_size == 0);
    static_assert(bs * element_size * 2 == sizeof(__m256i));
    std::array<__m256i, bs / 2> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* lo = reinterpret_cast<const __m128i*>(a + lda * (i + 0));
      auto* hi = reinterpret_cast<const __m128i*>(a + lda * (i + bs / 2));
      last_transpose[i] = _mm256_loadu2_m128i(hi, lo);
    }

    last_transpose =
        UnpackSequence<element_size, /*step_size=*/1, /*max_step_size=*/4>(
            last_transpose);

    if constexpr (element_size <= 8) {
      XLA_UNROLL
      for (int i = 0; i < bs / 2; ++i) {
#if defined(__AVX2__)
        last_transpose[i] = _mm256_permute4x64_epi64(last_transpose[i],
                                                     _MM_SHUFFLE(3, 1, 2, 0));
#else
        auto a = last_transpose[i];
        auto hi = _mm256_permute2f128_si256(a, a, 0b0001'0001);
        auto lo = _mm256_insertf128_si256(a, _mm256_castsi256_si128(a), 1);
        last_transpose[i] = _mm256_castpd_si256(_mm256_shuffle_pd(
            _mm256_castsi256_pd(lo), _mm256_castsi256_pd(hi), 0b1100));
#endif
      }
    }

    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      auto* lo = reinterpret_cast<__m128i*>(b + ldb * (i * 2 + 0));
      auto* hi = reinterpret_cast<__m128i*>(b + ldb * (i * 2 + 1));
      _mm256_storeu2_m128i(hi, lo, last_transpose[i]);
    }
  }
};
#endif

#if defined(XLA_HAS_SSE2)
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    std::array<__m128i, 4> loads;
    // [  0,  1,  2,  3 ]
    // [  4,  5,  6,  7 ]
    // [  8,  9, 10, 11 ]
    // [ 12, 13, 14, 15 ]
    XLA_UNROLL
    for (int i = 0; i < 4; ++i) {
      // Note: We would ideally use `_mm_loadu_si32` here but older compilers do
      // not support it. However, we can replicate it using a sequence such that
      // even older compilers will turn this into a single movd instruction.
      // memcpy is used because `a + lda * i` is not guaranteed to be aligned to
      // a 4-byte address.
      int load;
      memcpy(&load, a + lda * i, sizeof(load));
      loads[i] = _mm_cvtsi32_si128(load);
    }
    // [  0,  4,  1,  5,  2,  6,  3,  7 ]
    __m128i x_0_1 = _mm_unpacklo_epi8(loads[0], loads[1]);
    // [  8, 12,  9, 13, 10, 14, 11, 15 ]
    __m128i x_2_3 = _mm_unpacklo_epi8(loads[2], loads[3]);
    // [  0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15 ]
    __m128i x = _mm_unpacklo_epi16(x_0_1, x_2_3);

    // Note: We would ideally use `_mm_storeu_si32` here but older compilers do
    // not support it. However, we can replicate it using a sequence such that
    // even older compilers will turn this into a single movd instruction.
    // memcpy is used because `b + ldb * i` is not guaranteed to be aligned to a
    // 4-byte address.

    // [  0,  4,  8, 12 ]
    memcpy(b + ldb * 0, &x, sizeof(uint32_t));
    // [  1,  5,  9, 13 ]
    __m128i r1 = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 1, 1, 1));
    memcpy(b + ldb * 1, &r1, sizeof(uint32_t));
    // [  2,  6, 10, 14 ]
    __m128i r2 = _mm_unpackhi_epi32(x, x);
    memcpy(b + ldb * 2, &r2, sizeof(uint32_t));
    // [  3,  7, 11, 15 ]
    __m128i r3 = _mm_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 3, 3));
    memcpy(b + ldb * 3, &r3, sizeof(uint32_t));
  }
};
#endif

#ifdef XLA_HAS_SSE2
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using T = uint8_t;
    constexpr int bs = 8;
    constexpr size_t element_size = sizeof(T);
    // To help understand each step, let's show the contents of our SIMD
    // vectors.
    // The numbers shown are in octal and represent the source position from the
    // input in (row, column) format.
    //
    // [00, 01, 02, 03, 04, 05, 06, 07],
    // [10, 11, 12, 13, 14, 15, 16, 17],
    // [20, 21, 22, 23, 24, 25, 26, 27],
    // [30, 31, 32, 33, 34, 35, 36, 37],
    // [40, 41, 42, 43, 44, 45, 46, 47],
    // [50, 51, 52, 53, 54, 55, 56, 57],
    // [60, 61, 62, 63, 64, 65, 66, 67],
    // [70, 71, 72, 73, 74, 75, 76, 77],
    __m128i loads[bs];
    XLA_UNROLL
    for (int i = 0; i < bs; ++i) {
      loads[i] = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + lda * i));
    }

    // Pack adjacent loads together into one SIMD vector by interleaving the
    // lanes.
    //
    // [00, 10, 01, 11, 02, 12, 03, 13, 04, 14, 05, 15, 06, 16, 07, 17],
    // [20, 30, 21, 31, 22, 32, 23, 33, 24, 34, 25, 35, 26, 36, 27, 37],
    // [40, 50, 41, 51, 42, 52, 43, 53, 44, 54, 45, 55, 46, 56, 47, 57],
    // [60, 70, 61, 71, 62, 72, 63, 73, 64, 74, 65, 75, 66, 76, 67, 77],
    // In effect, we are splitting each SIMD vector into two blocks of 8
    // elements, then interleaving the elements.
    std::array<__m128i, bs / 2> last_transpose;
    XLA_UNROLL
    for (int i = 0; i < bs / 2; ++i) {
      // There is no need for _mm_unpackhi_epi8 as the high half of the two
      // vectors contains zeros.
      last_transpose[i] = _mm_unpacklo_epi8(loads[i * 2], loads[i * 2 + 1]);
    }

    // [00, 10, 20, 30, 40, 50, 60, 70, 01, 11, 21, 31, 41, 51, 61, 71],
    // [02, 12, 22, 32, 42, 52, 62, 72, 03, 13, 23, 33, 43, 53, 63, 73],
    // [04, 14, 24, 34, 44, 54, 64, 74, 05, 15, 25, 35, 45, 55, 65, 75],
    // [06, 16, 26, 36, 46, 56, 66, 76, 07, 17, 27, 37, 47, 57, 67, 77],
    last_transpose =
        UnpackSequence<element_size * 2, /*step_size=*/1, /*max_step_size=*/4>(
            last_transpose);

    // We have two rows stored in our 128-bit SIMD vector but our block size
    // is 64-bit, unpack and do two stores.
    //
    // [00, 10, 20, 30, 40, 50, 60, 70],
    // [01, 11, 21, 31, 41, 51, 61, 71],
    // [02, 12, 22, 32, 42, 52, 62, 72],
    // [03, 13, 23, 33, 43, 53, 63, 73],
    // [04, 14, 24, 34, 44, 54, 64, 74],
    // [05, 15, 25, 35, 45, 55, 65, 75],
    // [06, 16, 26, 36, 46, 56, 66, 76],
    // [07, 17, 27, 37, 47, 57, 67, 77],
    XLA_UNROLL
    for (int i = 0; i < 8; i += 2) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(b + ldb * (i + 0)),
                       last_transpose[i / 2]);
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(b + ldb * (i + 1)),
          _mm_unpackhi_epi64(last_transpose[i / 2], last_transpose[i / 2]));
    }
  }
};
#endif

// TODO(phawkins): Eigen doesn't have a SSE/AVX byte Packet16c type. Add one
// and call it here rather than using AVX intrinsics.
#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint8_t, 16>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_SSE2)
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    Sse2SquareTransposeMicroKernelImpl<uint8_t, /*bs=*/16>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/32> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint8_t, /*bs=*/32>::Apply(a, lda, b,
                                                                 ldb);
  }
};
#endif

#ifdef XLA_HAS_SSE2
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    // Note, SSE vectors can hold 8 uint16_t elements but our block size is 4.
    // We need to issue 4 loads to properly handle strides.
    //
    // [ 0,  1,  2,  3],
    // [ 4,  5,  6,  7],
    // [ 8,  9, 10, 11],
    // [12, 13, 14, 15],
    __m128i loads[4];
    XLA_UNROLL for (int i = 0; i < 4; ++i) {
      loads[i] = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + lda * i));
    }

    // [ 0,  4,  1,  5,  2,  6,  3,  7 ]
    auto unpack_16_0 = _mm_unpacklo_epi16(loads[0], loads[1]);
    // [ 8, 12,  9, 13, 10, 14, 11, 15 ]
    auto unpack_16_1 = _mm_unpacklo_epi16(loads[2], loads[3]);

    // Note: there is no need for _mm_unpackhi_epi16 as we only populate the
    // bottom 4 lanes.

    // [ 0,  4,  8, 12,  1,  5,  9, 13 ]
    auto unpack_32_0 = _mm_unpacklo_epi32(unpack_16_0, unpack_16_1);
    // [ 2,  6, 10, 14,  3,  7, 11, 15 ]
    auto unpack_32_1 = _mm_unpackhi_epi32(unpack_16_0, unpack_16_1);

    // [ 0,  4,  8, 12 ]
    _mm_storel_epi64(reinterpret_cast<__m128i*>(b + ldb * 0), unpack_32_0);
    // [ 1,  5,  9, 13 ]
    _mm_storel_epi64(reinterpret_cast<__m128i*>(b + ldb * 1),
                     _mm_unpackhi_epi64(unpack_32_0, unpack_32_0));
    // [ 2,  6, 10, 14 ]
    _mm_storel_epi64(reinterpret_cast<__m128i*>(b + ldb * 2), unpack_32_1);
    // [ 3,  7, 11, 15 ]
    _mm_storel_epi64(reinterpret_cast<__m128i*>(b + ldb * 3),
                     _mm_unpackhi_epi64(unpack_32_1, unpack_32_1));
  }
};
#endif

#if defined(__AVX__)
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint16_t, 8>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_SSE2)
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    Sse2SquareTransposeMicroKernelImpl<uint16_t, /*bs=*/8>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/16> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint16_t, /*bs=*/16>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxRectangularTransposeMicroKernelImpl<uint32_t, 4>::Apply(a, lda, b, ldb);
  }
};
#elif defined(XLA_HAS_SSE2)
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    Sse2SquareTransposeMicroKernelImpl<uint32_t, /*bs=*/4>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint32_t, /*bs=*/8>::Apply(a, lda, b,
                                                                 ldb);
  }
};
#endif

#ifdef XLA_HAS_SSE2
template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/2> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    Sse2SquareTransposeMicroKernelImpl<uint64_t, /*bs=*/2>::Apply(a, lda, b,
                                                                  ldb);
  }
};
#endif

#ifdef __AVX__
template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    AvxSquareTransposeMicroKernelImpl<uint64_t, /*bs=*/4>::Apply(a, lda, b,
                                                                 ldb);
  }
};
#endif  // __AVX__

#pragma pop_macro("XLA_ALWAYS_INLINE")
#pragma pop_macro("XLA_UNROLL")

}  // namespace xla

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_H_
