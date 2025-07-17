# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantized matmul kernel."""
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.quantized_matmul.tuned_block_sizes import get_tuned_block_sizes
import jax.numpy as jnp


def _quantize_array(
    x: jax.Array,  # [bs_block_size, in_block_size]
    x_abs_max_val: jax.Array,  # [1, bs_block_size]
):
  n_bits = 8
  int_max = 2 ** (n_bits - 1) - 1
  scale = (x_abs_max_val / int_max).T  # [bs_block_size, 1]
  x_int = jnp.round(x / scale).astype(jnp.int8)
  return x_int, scale.astype(jnp.float32)


def unfold_args(
    args: tuple[jax.Array | bool, ...], fn_args: tuple[bool, ...], fn
):
  """Minimize run-time branching by converting jax.Array bool to python bool."""
  if len(args) == 0:
    fn(*fn_args)
  else:
    arg = args[0]
    if isinstance(arg, bool):
      unfold_args(args[1:], fn_args + (arg,), fn)
    else:
      assert arg.dtype == jnp.bool and arg.size == 1
      lax.cond(
          arg,
          lambda: unfold_args(args[1:], fn_args + (True,), fn),
          lambda: unfold_args(args[1:], fn_args + (False,), fn),
      )


def matmul_kernel(
    x_ref: jax.Array,  # (batch_block_size, in_block_size)
    w_ref: jax.Array,  # (out_block_size, in_block_size)
    w_scale_ref: jax.Array,  # (1, out_block_size)
    x_abs_max_ref: jax.Array,  # (1, batch_block_size)
    out_ref: jax.Array,  # (batch_block_size, out_block_size)
    acc_scratch: jax.Array,  # (batch_block_size, out_block_size)
    q_x_scratch: jax.Array,  # (batch_block_size, in_block_size)
    x_scale_scratch: jax.Array,  # (batch_block_size, 1)
    *,
    quantize_activation: bool,
    save_acc: bool,
    save_q_x: bool,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
):
  bs_idx, out_idx, in_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  n_in = pl.num_programs(2)
  x_ref_dtype = x_ref.dtype
  assert x_ref.shape == (
      batch_block_size,
      in_block_size,
  ), 'x_ref shape is not correct'
  assert w_ref.shape == (
      out_block_size,
      in_block_size,
  ), 'w_ref shape is not correct'
  assert w_scale_ref.shape == (
      1,
      out_block_size,
  ), 'scalar_ref shape is not correct'
  assert x_abs_max_ref.shape == (
      1,
      batch_block_size,
  ), 'x_max_val shape is not correct'
  assert out_ref.shape == (
      batch_block_size,
      out_block_size,
  ), 'out_ref shape is not correct'

  if save_q_x:
    assert quantize_activation
    assert q_x_scratch is not None
    assert x_scale_scratch is not None
    quant = out_idx == 0
  else:
    assert q_x_scratch is None
    assert x_scale_scratch is None
    quant = quantize_activation

  if save_acc:
    assert acc_scratch is not None
    is_first_step = in_idx == 0
    is_last_step = in_idx == n_in - 1
  else:
    assert acc_scratch is None
    is_first_step = True
    is_last_step = True

  def matmul_body(quant, is_first_step, is_last_step):
    if quantize_activation:
      if quant:
        q_x_tmp, x_scale_tmp = _quantize_array(x_ref[...], x_abs_max_ref[...])
        if save_q_x:
          q_x_scratch[...] = q_x_tmp
          x_scale_scratch[...] = x_scale_tmp
      else:
        assert save_q_x
        q_x_tmp = q_x_scratch[...]
        if is_last_step:
          x_scale_tmp = x_scale_scratch[...]

      acc = jax.lax.dot_general(
          q_x_tmp,
          w_ref[...],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.int32,
      )
    else:
      acc = jax.lax.dot_general(
          x_ref[...],
          w_ref[...],
          (((1,), (1,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    if not is_first_step:
      acc += acc_scratch[...]

    if is_last_step:
      acc *= w_scale_ref[...]
      if quantize_activation:
        acc *= x_scale_tmp
      out_ref[...] = acc.astype(x_ref_dtype)
    else:
      assert save_acc
      acc_scratch[...] = acc

  unfold_args((quant, is_first_step, is_last_step), (), matmul_body)


def _next_multiple(x, multiple):
  return ((x + multiple - 1) // multiple) * multiple


@functools.partial(
    jax.jit,
    static_argnames=[
        'quantize_activation',
        'batch_block_size',
        'out_block_size',
        'in_block_size',
    ],
)
def quantized_matmul_kernel(
    x: jax.Array,  # [bs, n_input_features]
    w: jax.Array,  # [n_output_features, n_input_features]
    w_scale: jax.Array,  # [n_output_features]
    zero_point: jax.Array | None = None,  # [n_output_features]
    quant_block_size:  int | None = None,
    quantize_activation: bool = False,
    *,
    batch_block_size: int | None = None,
    out_block_size: int | None = None,
    in_block_size: int | None = None,
):
  assert zero_point is None, 'Not implemented: zero_point is not supported.'
  assert (
      quant_block_size is None
  ), 'Not implemented: quant_block_size is not supported.'

  # Pallas kernel only has access to a single block of the input. Therefere, for
  # per-token quantization, abs max has to be computed outside of the kernel.
  x_abs_max_val = jnp.max(jnp.abs(x), axis=-1, keepdims=False)  # [bs]
  # Pallas requires minormost dim to be a multiple of sublane size 128.
  # Therefore, instead of using [bs, 1], we reshape this into [1, bs]
  x_abs_max_val = jnp.expand_dims(x_abs_max_val, axis=0)  # [1, bs]
  assert x_abs_max_val.shape == (1, x.shape[0])

  orig_bs, orig_in_features = x.shape
  orig_out_features, _ = w.shape

  if (
      batch_block_size is None
      or out_block_size is None
      or in_block_size is None
  ):
    batch_block_size, out_block_size, in_block_size = get_tuned_block_sizes(
        orig_bs,
        orig_out_features,
        orig_in_features,
        jnp.dtype(x.dtype).name,
        quantize_activation,
    )

  padded_bs = _next_multiple(orig_bs, batch_block_size)
  if orig_bs < padded_bs:
    x = jnp.pad(x, ((0, padded_bs - orig_bs), (0, 0)))
    x_abs_max_val = jnp.pad(x_abs_max_val, ((0, 0), (0, padded_bs - orig_bs)))
  padded_out_features = _next_multiple(orig_out_features, out_block_size)
  if orig_out_features < padded_out_features:
    w = jnp.pad(w, ((0, padded_out_features - orig_out_features), (0, 0)))
    w_scale = jnp.pad(w_scale, (0, padded_out_features - orig_out_features))
  padded_in_features = _next_multiple(orig_in_features, in_block_size)
  if orig_in_features < padded_in_features:
    x = jnp.pad(x, ((0, 0), (0, padded_in_features - orig_in_features)))
    w = jnp.pad(w, ((0, 0), (0, padded_in_features - orig_in_features)))

  if w_scale.dtype != jnp.float32:
    w_scale = w_scale.astype(jnp.float32)
  w_scale = jnp.expand_dims(w_scale, axis=0)  # [1, n_output_features]

  assert (
      x.shape[1] == w.shape[1]
  ), f'{x.shape[1]=}) must be equal to {w.shape[1]=}'
  assert (
      w.shape[0] == w_scale.shape[1]
  ), f'{w.shape[0]=} must be equal to {w_scale.shape[1]=}'
  assert x_abs_max_val.shape == (
      1,
      x.shape[0],
  ), f'{x_abs_max_val.shape=} must be equal to (1, {x.shape[0]=})'
  assert (
      x.shape[0] % batch_block_size == 0
  ), f'{x.shape[0]=}) must be a multiple of block size {batch_block_size=}'
  assert (
      w.shape[0] % out_block_size == 0
  ), f'{w.shape[0]=} must be a multiple of block size {out_block_size=}'
  assert (
      x.shape[1] % in_block_size == 0
  ), f'{x.shape[1]=} must be a multiple of block size {in_block_size=}'

  acc_dtype = jnp.int32 if quantize_activation else jnp.float32
  vmem_to_be_transferred = (
      2
      * (
          batch_block_size * in_block_size * x.dtype.itemsize
          + out_block_size * in_block_size * w.dtype.itemsize
          + out_block_size * w_scale.dtype.itemsize
          + batch_block_size * x_abs_max_val.dtype.itemsize
          + batch_block_size * out_block_size * x.dtype.itemsize
      )
      + batch_block_size * out_block_size * jnp.dtype(acc_dtype).itemsize
  )
  # Within the kernel, it will use some extra VMEM for computation or vreg spills.
  vmem_used = vmem_to_be_transferred * 2
  vmem_limit_bytes = min(vmem_used * 2, 96 * 1024 * 1024)

  n_bs = padded_bs // batch_block_size
  n_out = padded_out_features // out_block_size
  n_in = padded_in_features // in_block_size

  save_acc = n_in > 1
  # Remove redundant input quantization logic by caching quantized input.
  # For best performance, only enable this behavior when single input block is used per batch.
  save_q_x = quantize_activation and n_in == 1 and n_out > 1

  kernel = pl.pallas_call(
      functools.partial(
          matmul_kernel,
          quantize_activation=quantize_activation,
          save_acc=save_acc,
          save_q_x=save_q_x,
          batch_block_size=batch_block_size,
          out_block_size=out_block_size,
          in_block_size=in_block_size,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec(
                  (batch_block_size, in_block_size), lambda b, o, i: (b, i)
              ),  # x
              pl.BlockSpec(
                  (out_block_size, in_block_size), lambda b, o, i: (o, i)
              ),  # w
              pl.BlockSpec(
                  (1, out_block_size), lambda b, o, i: (0, o)
              ),  # scalar
              pl.BlockSpec(
                  (1, batch_block_size), lambda b, o, i: (0, b)
              ),  # x_abs_max_val
          ],
          out_specs=pl.BlockSpec(
              (batch_block_size, out_block_size), lambda b, o, i: (b, o)
          ),
          scratch_shapes=[
              pltpu.VMEM((batch_block_size, out_block_size), acc_dtype)
              if save_acc
              else None,
              pltpu.VMEM((batch_block_size, in_block_size), jnp.int8)
              if save_q_x
              else None,
              pltpu.VMEM((batch_block_size, 1), jnp.float32)
              if save_q_x
              else None,
          ],
          grid=(n_bs, n_out, n_in),
      ),
      out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out_features), x.dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
  )

  kernel_name = f'quantized_matmul_int8_{batch_block_size}_{out_block_size}_{in_block_size}'
  # The named_scope is used for autotune. Different block sizes only impact the
  # pallas_call.
  with jax.named_scope(kernel_name):
    out = kernel(x, w, w_scale, x_abs_max_val)

  return out[:orig_bs, :orig_out_features]
