// Common code shared between CPU and CUDA.

#pragma once

#include <ATen/native/DispatchStub.h>

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

using check_grid_sampler_2arg_fn = void (*) (
  const TensorBase& input,
  const TensorBase& grid);

using check_grid_sampler_3arg_fn = void (*) (
  const TensorBase& input,
  const TensorBase& grid,
  GridSamplerInterpolation interpolation_mode);

using cond_cudnn_grid_sampler_fn = bool (*) (
  const TensorBase& input,
  const TensorBase& grid);

DECLARE_DISPATCH(check_grid_sampler_2arg_fn, check_grid_sampler_common_stub);
DECLARE_DISPATCH(check_grid_sampler_2arg_fn, check_grid_sampler_2d_stub);
DECLARE_DISPATCH(check_grid_sampler_3arg_fn, check_grid_sampler_3d_stub);
DECLARE_DISPATCH(cond_cudnn_grid_sampler_fn, cond_cudnn_grid_sampler_stub);

}} // namespace at::native