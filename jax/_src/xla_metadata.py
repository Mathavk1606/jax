# Copyright 2024 The JAX Authors.
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

from typing import Any

from jax._src import config
from jax._src import core
from jax._src import tree_util
from jax._src import xla_metadata_lib
from jax._src.basearray import ArrayLike
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir

config_ext = xla_client._xla.config


class XlaMetadataContextManager:
  __slots__ = ["prev", "updates"]

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        xla_metadata_lib.update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)


def set_xla_metadata(
    value: ArrayLike | None = None, **kwargs
) -> XlaMetadataContextManager | ArrayLike:
  if value is None:
    return XlaMetadataContextManager(kwargs)
  else:
    hashable_metadata = tuple(sorted(kwargs.items()))
    value = tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(v, xla_metadata=hashable_metadata),
        value,
    )
    return value


# `xla_metadata_value_p` is an identity primitive for attaching frontend_attributes
# to the primitive's producing (parent/owner) op.

xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(lambda value, *, xla_metadata: value)
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata: aval)
# TODO(nbasile): Implement tagging gradient ops with metadata.
ad.deflinear2(xla_metadata_value_p, lambda ct, _: (ct,))


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, value_mlir: ir.Value, *, xla_metadata
):
  xla_metadata = dict(xla_metadata)
  op_to_attach_metadata = _target_op_to_attach_metadata(value_mlir)
  if op_to_attach_metadata:
    _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
  return [value_mlir]


mlir.register_lowering(
    xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False
)


def _xla_metadata_value_batching_rule(batched_args, bdims, *, xla_metadata):
  (value_batched,) = batched_args
  (value_bdim,) = bdims
  out_batched = xla_metadata_value_p.bind(
      value_batched, xla_metadata=xla_metadata
  )
  return out_batched, value_bdim


batching.primitive_batchers[xla_metadata_value_p] = (
    _xla_metadata_value_batching_rule
)


# -----------------------------------------------------------------------------


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
  op = value_mlir.owner
  if op is None or isinstance(op, ir.Block):
    return None
  # TODO(nbasile): Add logic for handling multiply-by-constant-1.0 ops, which
  # are often added by jax gradients.
  # [Couple this change with tagging gradient ops.]
  return op


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  ctx_attributes = {}
  existing_attributes = {}
  if xla_metadata:
    for k, v in xla_metadata.items():
      ctx_attributes[k] = ir.StringAttr.get(str(v).lower())
    # Combine with existing mhlo.frontend_attributes
    op_attributes_dict = {attr.name: attr.attr for attr in op.attributes}
    for k, attributes in op_attributes_dict.items():
      if k == "mhlo.frontend_attributes":
        v_dict = {attr.name: attr.attr for attr in attributes}
        for fa_key, fa_val in v_dict.items():
          existing_attributes[fa_key] = fa_val
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        ctx_attributes | existing_attributes
    )
