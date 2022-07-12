// Ternary and higher-order pointwise operations
#include <ATen/native/PointwiseOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/NamedTensorUtils.h>

namespace at {
namespace meta {

TORCH_META_FUNC(addcmul)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  build_ternary_op(maybe_get_output(), self, tensor1, tensor2);
}

TORCH_META_FUNC(addcdiv)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  build_ternary_op(maybe_get_output(), self, tensor1, tensor2);
}

} // namespace meta
namespace native {

TORCH_IMPL_FUNC(addcmul_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcmul_stub(device_type(), *this, value);
}

TORCH_IMPL_FUNC(addcdiv_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  addcdiv_stub(device_type(), *this, value);
}

DEFINE_DISPATCH(addcmul_stub);
DEFINE_DISPATCH(addcdiv_stub);

} // namespace native
} // namespace at
