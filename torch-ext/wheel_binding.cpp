#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "torch_binding.h"
// This file provides a standalone Torch extension binding suitable for
// building a pip wheel directly from this repository (without kernel-builder).
//
// It registers an operator under the `metal_flash_sdpa` namespace:
//   torch.ops.metal_flash_sdpa.flash_attention_varlen(...)
//
// The implementation is only provided for the MPS dispatch key.
namespace {
void flash_attention_varlen_wrapper(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool do_causal,
    double scale,
    double softcapping) {
  // Ensure we pass lvalues to the underlying implementation (which takes non-const refs).
  auto out_ = out;
  auto query_ = query;
  auto key_ = key;
  auto value_ = value;
  auto cu_q_ = cu_seqlens_q;
  auto cu_k_ = cu_seqlens_k;
  flash_attention_varlen(
      out_,
      query_,
      key_,
      value_,
      cu_q_,
      cu_k_,
      max_seqlen_q,
      max_seqlen_k,
      do_causal,
      scale,
      softcapping);
}
} // namespace
TORCH_LIBRARY(metal_flash_sdpa, m) {
  m.def(
      "flash_attention_varlen(Tensor! out, Tensor query, Tensor key, Tensor value, Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, bool do_causal, float scale, float softcapping) -> ()");
}
TORCH_LIBRARY_IMPL(metal_flash_sdpa, MPS, m) {
  m.impl("flash_attention_varlen", flash_attention_varlen_wrapper);
}

// Define the Python module entrypoint so this can be imported as `metal_flash_sdpa._C`.
// All functionality is exposed via `torch.ops.metal_flash_sdpa.*` after import.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Metal Flash-SDPA Torch extension (MPS).";
}

