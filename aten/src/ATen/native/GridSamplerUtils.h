#pragma once

// See NOTE: [Tensor vs. TensorBase]
// https://github.com/pytorch/pytorch/pull/66979
namespace at {
class TensorBase;
}

namespace at { namespace native {

namespace detail {

  enum class GridSamplerInterpolation {Bilinear, Nearest, Bicubic};
  enum class GridSamplerPadding {Zeros, Border, Reflection};

}  // namespace detail

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

void check_grid_sampler_common(const TensorBase& input, const TensorBase& grid);

void check_grid_sampler_2d(const TensorBase& input, const TensorBase& grid);

void check_grid_sampler_3d(
  const TensorBase& input,
  const TensorBase& grid,
  GridSamplerInterpolation interpolation_mode
);

bool cond_cudnn_grid_sampler(const TensorBase& input, const TensorBase& grid);

}}