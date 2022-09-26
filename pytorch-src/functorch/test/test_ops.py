# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import functools
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_device_type import \
    toleranceOverride, tol
from functorch_additional_op_db import additional_op_db
from torch.testing._internal.common_methods_invocations import op_db
from common_utils import (
    get_fallback_and_vmap_exhaustive,
    get_exhaustive_batched_inputs,
    get_exhaustive_batched_inputs_batch_norm_is_training,
    xfail,
    skip,
    skipOps,
    tol1,
    # tol2,
    opsToleranceOverride,
    check_vmap_fallback,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    loop2,
)

from torch.testing._internal.opinfo.core import SampleInput
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from functorch import grad, vjp, vmap, jacrev, jacfwd
import torch.autograd.forward_ad as fwAD
from functorch._src.eager_transforms import _as_tuple, jvp
aten = torch.ops.aten


# Version of autograd.grad with some differences:
#   - pytree inputs is allowed (but leaves of the pytree have to all
#     be tensors)
#   - if an input is not used as part of derivatives, we will return a
#     zero-filled tensor for the result
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    inputs, inputs_spec = tree_flatten(inputs)
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        diff_inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all([is_differentiable_arg(a) for a in arg]):
            return True
        if all([not is_differentiable_arg(a) for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(i for i, arg in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


# TODO: consolidate with normalize_op_input_output2
def normalize_op_input_output3(f, args, kwargs, sample_args, output_process_fn_grad=None):
    flat_args, args_spec = tree_flatten(args)
    flat_sample_args, _ = tree_flatten(sample_args)
    diff_argnums = tuple(i for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
                         if diff_arg(sample, requires_grad=True))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad
    )


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


def simulate_jvp(f, primals, tangents):
    primals_out, tangents_out = torch.autograd.functional.jvp(f, primals, tangents)
    return primals_out, tangents_out


def ref_jvp(f, primals, tangents):
    with fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        result_duals = f(*duals)
        result_duals, spec = tree_flatten(result_duals)
        primals_out, tangents_out = zip(*(fwAD.unpack_dual(d) for d in result_duals))
        return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)


def get_sample_cotangents(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    output = fn(*primals)
    return tree_map(torch.randn_like, output)


# returns a new function g(*args, *cotangents)
# that computes vjps and (*args, cotangents)
def get_vjp_fn_and_args_with_cotangents(f, sample, cotangents):
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        assert len(args) == len(flat_args) + len(flat_cotangents)
        actual_args = args[:len(flat_args)]
        cotangents = args[len(flat_args):]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(f, actual_args, kwargs,
                                                 flat_args,
                                                 sample.output_process_fn_grad)
        _, vjp_fn = vjp(fn, *primals)
        return vjp_fn(cotangents)

    return wrapped, tuple(flat_args + flat_cotangents)


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result))
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        result, vjp_fn = vjp(fn, *primals)
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        return vjp_fn(cotangents)

    return wrapped, args


def get_jvp_variant(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        tangents = args
        primals_out, tangents_out = jvp(fn, primals, tangents)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, tangents


def get_jvp_variant_primals_tangents(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        primals_in = args[:len(primals)]
        tangents_in = args[len(primals):]
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    xfail('tensor_split'),  # data_ptr composite compliance
}


class TestOperators(TestCase):
    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_grad', vjp_fail.union({
        xfail('linalg.eig'),  # diagonal_scatter does not support complex
        xfail('chalf', '', device_type='cpu'),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
        skip('as_strided_scatter', ''),  # silent incorrectness; seems flaky
        xfail('sparse.sampled_addmm', ''),  # RuntimeError: Sparse CSR tensors do not have strides
        xfail('to_sparse', ''),  # Could not run 'aten::sum.dim_IntList'
    }))
    @opsToleranceOverride('TestOperators', 'test_grad', (
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
    ))
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped for redundancy. test_vjp handles in-place testing.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return result.sum()
                result = sum([res.sum() for res in result])
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)

            self.assertEqual(result, expected)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_jvp', set({
        # Composite ops that do bad things. Need to be fixed in PyTorch core.
        # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
        xfail('tensor_split'),

        # BUG: silent incorrectness: runs and produces numerical differences
        skip('nn.functional.max_unpool1d'),  # fails everywhere except on mac
        skip('nn.functional.max_unpool2d'),  # fails everywhere except on windows
        skip('nn.functional.max_unpool3d'),  # fails everywhere except on mac

        xfail('nn.functional.rrelu')  # in-place test errors out with no formula implemented
    }))
    @opsToleranceOverride('TestOperators', 'test_jvp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=1e-04, rtol=1.3e-06)}, device_type='cuda'),
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=4e-04, rtol=4e-04)}),
    ))
    def test_jvp(self, device, dtype, op):
        # TODO: get rid of vjp_decomp when we add decomposition support to
        # PyTorch's forward-mode ad. Currently the decomposition support only
        # works for functorch.jvp
        VJP_DECOMP = {
            'nn.functional.logsigmoid',
        }
        if op.name in VJP_DECOMP:
            fixme_ref_jvp_local = simulate_jvp
        else:
            fixme_ref_jvp_local = ref_jvp

        if not op.supports_forward_ad and op.name not in VJP_DECOMP:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        outplace_variant = op if not is_inplace(op, op.get_op()) else None
        inplace_variant = op.inplace_variant if op.supports_inplace_autograd else None

        for sample in samples:
            args = (sample.input,) + sample.args
            kwargs = sample.kwargs
            if outplace_variant:
                self.jvp_opinfo_test(outplace_variant, args, kwargs,
                                     sample.output_process_fn_grad,
                                     clone_inputs=False,
                                     fixme_ref_jvp_local=fixme_ref_jvp_local)
            if is_valid_inplace_sample_input(sample, op, inplace_variant):
                self.jvp_opinfo_test(inplace_variant, args, kwargs,
                                     sample.output_process_fn_grad,
                                     clone_inputs=True,
                                     fixme_ref_jvp_local=fixme_ref_jvp_local)

    def jvp_opinfo_test(self, fn, args, kwargs, output_process_fn,
                        clone_inputs, fixme_ref_jvp_local):
        # NB: we used requires_grad=True to determine where the primals are,
        # but don't need that information otherwise
        fn, primals = normalize_op_input_output2(
            fn, args, kwargs, output_process_fn, requires_grad=True)
        orig_primals = tree_map(lambda x: x.detach(), primals)
        orig_tangents = tree_map(lambda x: torch.randn_like(x), primals)

        def maybe_clone_inputs():
            if clone_inputs:
                primals = tree_map(torch.clone, orig_primals)
                tangents = tree_map(torch.clone, orig_tangents)
                return primals, tangents
            return orig_primals, orig_tangents

        primals, tangents = maybe_clone_inputs()
        expected_primal_outs, expected_tangent_outs = \
            fixme_ref_jvp_local(fn, primals, tangents)

        primals, tangents = maybe_clone_inputs()
        primal_outs, tangent_outs = jvp(fn, primals, tangents)

        self.assertEqual(primal_outs, expected_primal_outs)
        self.assertEqual(tangent_outs, expected_tangent_outs)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjp', vjp_fail.union({
        skip('as_strided_scatter', ''),  # silent incorrectness; also might be flaky
        xfail('sparse.sampled_addmm', ''),
    }))
    @opsToleranceOverride('TestOperators', 'test_vjp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=5e-05, rtol=9e-05)}, device_type='cuda'),
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
    ))
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def _test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(sample, op, op.inplace_variant):
                    continue
                fn, primals = normalize_op_input_output(_op, sample)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                out, vjp_fn = vjp(fn, *primals)
                self.assertEqual(out, result)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

        _test(op)
        for a_op in op.aliases:
            _test(a_op)
        if op.inplace_variant:
            def f(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)
            _test(f, inplace=True)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvjp', vjp_fail.union({
        skip('nn.functional.max_unpool1d'),  # silent incorrectness; Flaky
        skip('nn.functional.max_unpool2d'),  # silent incorrectness; Flaky
        xfail('nn.functional.ctc_loss'),  # Not Implemented
        xfail('native_layer_norm', ''),  # Expected a proper Tensor but got None for argument #1 'other'
        xfail('sparse.sampled_addmm', ''),  # sparse tensors have no strides
        # AssertionError: Tensor-likes are not close!
        # Mismatched elements: 1 / 15 (6.7%)
        # Greatest absolute difference: 24.0 at index (2, 4) (up to 1e-05 allowed)
        # Greatest relative difference: 1.7933241714393998e-06 at index (2, 4) (up to 1.3e-06 allowed)
        # The failure occurred for item [0]
        xfail('_masked.prod')
    }))
    @opsToleranceOverride('TestOperators', 'test_vjpvjp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=5e-05, rtol=9e-05)}, device_type='cuda'),
        tol1('prod',
             {torch.float32: tol(atol=2e-05, rtol=1e-04)}),
        tol1('_masked.cumprod',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('cumprod',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('linalg.vander',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
    ))
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        def test(_op, inplace=False):
            for sample in samples:
                if inplace and not is_valid_inplace_sample_input(sample, op, op.inplace_variant):
                    continue
                fn, args = get_vjpfull_variant(_op, sample)
                result = fn(*args)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                # Compute vjp of vjp
                _, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)

                # Compute ref_vjp of vjp. We could have done ref_vjp of ref_vjp,
                # but since we're confident that vjp works by itself, this is
                # an equivalent way to test that.
                _, vjp_fn = ref_vjp(fn, *args)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

        test(op)
        if op.inplace_variant:
            def fn(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)
            test(fn, inplace=True)

    @skipOps('TestOperators', 'test_vmapvjpvjp', vjp_fail.union({
        skip("atleast_1d"),  # Takes too long
        skip("atleast_2d"),  # Takes too long
        skip("atleast_3d"),  # Takes too long
        xfail("as_strided"),  # incorrect output
        xfail("as_strided_scatter"),  # incorrect output
        skip("bernoulli"),  # calls random op
        xfail("bfloat16"),  # rank 4 tensor for channels_last
        xfail("chalf"),  # rank 4 tensor for channels_last
        xfail("double"),  # rank 4 tensor for channels_last
        xfail("float"),  # rank 4 tensor for channels_last
        xfail("half"),  # rank 4 tensor for channels_last
        # It looks like you're either (1) calling .item() on a Tensor or
        # (2) attempting to use a Tensor in some data-dependent control flow or
        # (3) encountering this error in PyTorch internals.
        xfail("index_reduce"),
        xfail("linalg.eig"),  # vmap over torch.allclose
        xfail("linalg.eigvals"),  # vmap over torch.allclose
        xfail("linalg.householder_product"),  # vmap: inplace into a regular tensor
        xfail("nanquantile", device_type='cpu'),  # vmap not implemented for at::equal.
        xfail("native_layer_norm"),  # vmap: inplace into a regular tensor
        # got a batched tensor as input while the running_mean or running_var,
        # which will be updated in place, were not batched.
        xfail("nn.functional.batch_norm"),
        xfail("nn.functional.binary_cross_entropy"),  # vmap: inplace into a regular tensor
        xfail("nn.functional.ctc_loss"),  # derivate not implemented for _ctc_loss_backward
        skip("nn.functional.dropout"),  # calls random op
        skip("nn.functional.dropout2d"),  # calls random op
        skip("nn.functional.dropout3d"),  # calls random op
        skip("nn.functional.feature_alpha_dropout", "with_train"),  # calls random op
        skip("nn.functional.fractional_max_pool2d"),  # calls random op
        skip("nn.functional.fractional_max_pool3d"),  # calls random op
        skip('nn.functional._scaled_dot_product_attention'),  # randomness
        # It looks like you're either (1) calling .item() on a Tensor or
        # (2) attempting to use a Tensor in some data-dependent control flow or
        # (3) encountering this error in PyTorch internals.
        xfail("nn.functional.gaussian_nll_loss"),
        # got a batched tensor as input while the running_mean or running_var,
        # which will be updated in place, were not batched.
        xfail("nn.functional.instance_norm"),
        xfail("nn.functional.layer_norm"),  # vmap: inplace into a regular tensor
        # RuntimeError: NYI: querying is_contiguous inside of vmap
        # for memory_format other than torch.contiguous_formats
        xfail("nn.functional.max_pool2d"),
        # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
        # supported with memory_format torch.preserve_format or
        # torch.contiguous_format (got ChannelsLast)
        xfail("nn.functional.max_unpool2d"),
        # RuntimeError: NYI: Tensor.clone(memory_format) inside vmap is only
        # supported with memory_format torch.preserve_format
        # or torch.contiguous_format (got ChannelsLast)s
        xfail("nn.functional.max_unpool2d", "grad"),
        xfail("nn.functional.rrelu"),  # RuntimeError: vmap: we do not yet support aten::rrelu_with_noise.
        xfail("normal"),  # calls random op
        xfail("normal", "number_mean"),  # calls random op
        xfail("pca_lowrank"),  # calls random op
        xfail("put"),  # vmap: inplace into a regular tensor
        xfail("quantile", device_type='cpu'),  # Batching rule not implemented for `at::equal`
        xfail("scatter_reduce", "prod"),  # vmap (looks like you are calling item/data-dependent)
        xfail("sparse.sampled_addmm"),  # RuntimeError: Sparse CSR tensors do not have strides
        xfail("svd_lowrank"),  # calls random op
        xfail("take"),  # vmap: inplace into a regular tensor
        xfail("to"),  # rank 4 tensor for channels_last
        xfail("view_as_complex"),  # RuntimeError: Tensor must have a last dimension with stride 1
        xfail("_masked.softmax", device_type='cuda'),  # Mismatch in values!
        xfail("_masked.softmin", device_type='cuda'),  # Mismatch in values!
        # got a batched tensor as input while the running_mean or running_var,
        # which will be updated in place, were not batched.
        xfail("nn.functional.batch_norm", 'without_cudnn'),
        # view doesn't work on sparse
        xfail("to_sparse"),
    }))
    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride('TestOperators', 'test_vmapvjpvjp', (
        tol1('linalg.svd',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('linalg.lu_factor',
             {torch.float32: tol(atol=2e-03, rtol=2e-02)}),
        tol1('svd',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
    ))
    def test_vmapvjpvjp(self, device, dtype, op):
        # Since, we test `vjpvjp` independently,
        # for this test, we just verify that vmap
        # of `vjpvjp` is correct.
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents, _ = tree_flatten(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result, _ = tree_flatten(result)
                result_vjps, _ = tree_flatten(result_vjps)
                return (*result, *result_vjps)

            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                vjp_of_vjp, args_and_cotangents, {}, is_batch_norm_and_training=is_batch_norm_and_training)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapvjp_fail = vjp_fail.union({
        # -------------------- ALLOWED FAILURES --------------------------------
        # The following are not bugs and are expected behavior
        xfail('masked_select'),  # Not possible due to dynamic shapes
        skip('bernoulli'),  # randomness
        skip('normal', ''),  # randomness
        skip('normal', 'number_mean'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.feature_alpha_dropout', 'with_train'),  # randomness
        skip('nn.functional.feature_alpha_dropout', 'without_train'),  # randomness
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.dropout2d'),  # randomness
        skip('nn.functional.dropout3d', ''),  # randomness
        skip('nn.functional._scaled_dot_product_attention'),  # randomness
        xfail('as_strided'),  # as_strided is too wild for us to support, wontfix
        xfail('index_put', ''),  # not possible due to dynamic shapes; we support a subset
        xfail('masked_scatter'),  # dynamic
        xfail('nn.functional.fractional_max_pool2d'),  # random
        xfail('nn.functional.fractional_max_pool3d'),  # random
        xfail('take'),  # dynamic
        xfail('pca_lowrank', ''),  # randomness
        xfail('svd_lowrank', ''),  # randomness
        xfail('to_sparse', ''),  # non-dense output
        skip('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        # ----------------------------------------------------------------------

        # ---------------------------- BUGS ------------------------------------
        # All of the following are bugs and need to be fixed
        skip('linalg.svdvals'),  # # really annoying thing where it passes correctness check but not has_batch_rule
        xfail('__getitem__', ''),  # dynamic error
        xfail('linalg.eig'),  # Uses aten::allclose
        xfail('linalg.householder_product'),  # needs select_scatter
        xfail('nanquantile', device_type='cpu'),  # checks q via a .item() call
        xfail('nn.functional.gaussian_nll_loss'),  # checks var for if any value < 0
        xfail('narrow'),  # .item() call
        xfail('quantile', device_type='cpu'),  # checks q via a .item() call
        xfail('view_as_complex'),  # Tensor must have a last dimension with stride 1

        # required rank 4 tensor to use channels_last format
        xfail('bfloat16'),
        xfail('double'),
        xfail('float'),
        xfail('half'),
        xfail('chalf', ''),

        xfail('scatter_reduce', 'prod'),  # item call

        # Batching rule not implemented for aten::_use_cudnn_ctc_loss.Tensor
        xfail('nn.functional.ctc_loss', device_type='cuda'),
        # NYI: querying is_contiguous inside of vmap for memory_format other than torch.contiguous_format
        xfail('nn.functional.max_unpool2d'),
        xfail('nn.functional.max_unpool2d', 'grad'),

        xfail('sparse.sampled_addmm', ''),
        xfail('as_strided_scatter', ''),  # calls as_strided
        xfail('index_reduce', ''),  # .item() call
        # ---------------------------------------------------------------------
    })

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride('TestOperators', 'test_vmapvjp', (
        tol1('linalg.svd',
             {torch.float32: tol(atol=1.5e-04, rtol=1e-04)}, device_type="cuda"),
        tol1('svd',
             {torch.float32: tol(atol=1.5e-04, rtol=1e-04)}, device_type="cuda"),
    ))
    @skipOps('TestOperators', 'test_vmapvjp', vmapvjp_fail)
    def test_vmapvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            cotangents = get_sample_cotangents(op, sample)
            fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapjvpall_fail = {
        # -------------------- ALLOWED FAILURES --------------------------------
        # The following are expected (not a bug)
        skip('bernoulli', ''),  # randomness
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.dropout2d', ''),
        skip('nn.functional.dropout3d', ''),
        skip('nn.functional._scaled_dot_product_attention'),  # randomness
        skip('nn.functional.feature_alpha_dropout', 'without_train'),
        skip('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('nn.functional.fractional_max_pool2d'),  # Cannot access data pointer of Tensor that doesn't have storage
        xfail('nn.functional.fractional_max_pool3d'),  # Cannot access data pointer of Tensor that doesn't have storage
        # Not actually a problem: embedding with max_norm mutates the weight
        # and causes different runs to produce different results.
        # skip because this is flaky depending on what the max_norm is!
        skip('nn.functional.embedding', ''),
        skip('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        # ----------------------------------------------------------------------

        # ---------------------------- BUGS ------------------------------------
        # The following are bugs that we should fix
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs on cuda
        xfail('_masked.mean'),  # silent incorrectness (nan difference)

        xfail('nn.functional.soft_margin_loss', ''),  # soft_margin_loss_backward does not support forward-ad
        xfail('tensor_split'),  # data_ptr composite compliance
        xfail('quantile'),  # at::equal batching rule (cpu), also, in-place vmap (cuda)
        skip('as_strided'),  # Test runner cannot handle this
        xfail('nn.functional.gaussian_nll_loss'),  # .item or data-dependent control flow
        xfail('scatter'),  # forward-mode AD does not support at::scatter
        xfail('nanquantile'),  # at::equal batching rule (cpu), also, in-place vmap (cuda)
        xfail('view_as_complex'),  # Tensor must have a last dimension with stride 1

        skip('pca_lowrank', ''),  # randomness
        skip('svd_lowrank', ''),  # randomness

        xfail('double'),  # required rank 4 tensor to use channels_last format

        # potential silent incorrectness
        skip('nn.functional.max_unpool1d'),  # Flaky, seems to sometimes his max_unpool2d
        skip('nn.functional.max_unpool2d'),  # fails everywhere except on mac
        skip('nn.functional.max_unpool3d'),  # fails everywhere except on mac

        # erroring because running_mean and running_var aren't differentiable
        xfail('nn.functional.batch_norm'),
        xfail('nn.functional.batch_norm', 'without_cudnn'),
        # ----------------------------------------------------------------------
    }

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @opsToleranceOverride('TestOperators', 'test_vmapjvpall', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=2e-04, rtol=9e-3)}, device_type='cuda'),
        tol1('linalg.householder_product',
             {torch.float32: tol(atol=2e-04, rtol=9e-3)}, device_type='cuda'),
        tol1('linalg.householder_product',
             {torch.float32: tol(atol=2e-04, rtol=1e-4)}, device_type='cpu'),
    ))
    @skipOps('TestOperators', 'test_vmapjvpall', vmapjvpall_fail)
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    # This is technically a superset of test_vmapjvp. We should either delete test_vmapjvp
    # or figure out if we can split vmapjvpall. It's useful to keep test_vmapjvp intact
    # because that coresponds to "batched forward-mode AD" testing in PyTorch core
    def test_vmapjvpall(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple(arg_values) + tuple(kwarg_values)
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            is_batch_norm_and_training = is_batch_norm_training(op.name, kwarg_values)
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapjvpall_has_batch_rule', vmapjvpall_fail.union({
        skip('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        xfail('nn.functional.huber_loss'),
        xfail('lu'),
        xfail('cumprod'),
        xfail('masked_fill'),
        xfail('copysign'),
        xfail('complex'),
        skip('_masked.mean'),  # ???
        xfail('masked_scatter'),
        xfail('index_fill'),
        xfail('put'),
        xfail('take'),
        xfail('nn.functional.max_pool3d'),
        xfail('vdot'),
        xfail('nanmean'),
        xfail('nansum'),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('linalg.lu_factor', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('pca_lowrank', ''),
        xfail('svd_lowrank', ''),
        xfail('linalg.lu_factor_ex', ''),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('special.log_ndtr', ''),
        xfail('fft.ihfft2'),  # conj_physical fallback
        xfail('fft.ihfftn'),  # conj_physical fallback
        xfail('istft'),  # col2im fallback
        xfail('polar'),  # complex fallback
        xfail('nn.functional.max_unpool3d', 'grad'),
        xfail('nn.functional.smooth_l1_loss', ''),
        xfail('nn.functional.max_unpool2d', 'grad'),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('nn.functional.max_unpool1d', 'grad'),
        xfail('nn.functional.embedding', ''),
        xfail('scatter_reduce', "sum"),   # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "mean"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "amin"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "amax"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('lu_unpack'),
        xfail('nn.functional.glu'),
        xfail('nn.functional.bilinear'),  # trilinear doesn't have batching rule
        xfail('linalg.lu', ''),
        xfail('linalg.lu_solve', ''),
        xfail('nn.functional.dropout3d', ''),
        xfail('as_strided_scatter', ''),
        xfail('_masked.cumprod', ''),
        xfail('linalg.vecdot', ''),
    }))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    def test_vmapjvpall_has_batch_rule(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        def test():
            for sample in samples:
                arg_values = [sample.input] + list(sample.args)
                kwarg_values = sample.kwargs
                args = tuple(arg_values) + tuple(kwarg_values)
                fn, args = get_jvp_variant_primals_tangents(op, sample)
                is_batch_norm_and_training = is_batch_norm_training(op.name, kwarg_values)
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training, compute_loop_out=False):
                    pass
        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @skipOps('TestOperators', 'test_vmapvjp_has_batch_rule', vmapvjp_fail.union({
        skip('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        xfail('view_as_complex'),
        xfail('complex'),
        xfail('copysign'),
        xfail('cummax'),
        xfail('cummin'),
        xfail('cumprod'),
        xfail('nansum'),
        xfail('nanmean'),
        xfail('narrow'),  # Batching rule not implemented for `narrow.Tensor` (and view op)
        xfail('special.log_ndtr'),
        xfail('index_copy'),
        xfail('index_fill'),
        xfail('linalg.eig'),
        xfail('linalg.householder_product'),
        xfail('lu'),
        xfail('lu_solve'),
        xfail('lu_unpack'),
        xfail('masked_fill'),
        xfail('masked_scatter'),
        xfail('masked_select'),
        xfail('nanquantile'),
        xfail('put'),
        xfail('scatter_reduce', "sum"),   # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "mean"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "amin"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('scatter_reduce', "amax"),  # aten::scatter_reduce.two hit the vmap fallback
        xfail('quantile'),
        xfail('renorm'),
        xfail('take'),
        xfail('tensor_split'),
        xfail('to_sparse'),
        xfail('unfold'),
        xfail('vdot'),
        xfail('nn.functional.dropout'),
        xfail('fft.ihfft2'),
        xfail('fft.ihfftn'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('nn.functional.huber_loss'),
        xfail('nn.functional.bilinear'),
        xfail('nn.functional.fractional_max_pool3d'),
        xfail('nn.functional.ctc_loss'),
        xfail('as_strided'),
        xfail('stft'),
        xfail('nn.functional.rrelu'),
        xfail('nn.functional.embedding_bag'),
        xfail('nn.functional.max_pool3d'),
        xfail('istft'),
        xfail('nn.functional.fractional_max_pool2d'),
        xfail('linalg.lu_factor', ''),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('pca_lowrank', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('svd_lowrank', ''),
        xfail('linalg.lu_factor_ex', ''),

        xfail('nn.functional.max_unpool2d', ''),
        xfail('nn.functional.multi_margin_loss', ''),
        xfail('nn.functional.multilabel_margin_loss', ''),
        xfail('nn.functional.pdist', ''),
        xfail('nn.functional.smooth_l1_loss', ''),
        xfail('scatter_reduce', 'prod'),
        xfail('nn.functional.max_unpool1d', ''),
        xfail('nn.functional.max_unpool3d', ''),
        xfail('nn.functional.max_unpool3d', 'grad'),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('nn.functional.max_unpool1d', 'grad'),
        xfail('nn.functional.max_unpool2d', 'grad'),
        xfail('linalg.lu', ''),
        xfail('linalg.lu_solve', ''),
        xfail('chalf', ''),
        xfail('index_reduce', ''),
        xfail('linalg.vander', ''),
        xfail('nn.functional.dropout3d', ''),
        xfail('as_strided_scatter', ''),
        xfail('segment_reduce', 'offsets'),
        xfail('_masked.cumprod', ''),
        xfail('linalg.vecdot', ''),
        xfail('segment_reduce', 'lengths'),
        xfail('sparse.sampled_addmm', ''),
    }))
    def test_vmapvjp_has_batch_rule(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        def test():
            for sample in samples:
                cotangents = get_sample_cotangents(op, sample)
                fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
                is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training, compute_loop_out=False):
                    pass
                for a_op in op.aliases:
                    fn, args = get_vjp_fn_and_args_with_cotangents(a_op, sample, cotangents)
                    for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                            fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training, compute_loop_out=False):
                        pass

        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvmap', vjp_fail.union({
        skip('bernoulli', ''),  # vjpvmap testing can't handle randomness
        skip('normal', ''),  # vjpvmap testing can't handle randomness
        skip('normal', 'number_mean'),  # vjpvmap testing can't handle randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.feature_alpha_dropout', 'with_train'),  # randomness
        skip('nn.functional.feature_alpha_dropout', 'without_train'),  # randomness
        skip('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        skip('to_sparse', ''),  # non-dense output

        # fallback path doesn't work
        # All of the following are bugs and need to be fixed
        xfail('__getitem__', ''),
        xfail('index_put', ''),
        xfail('view_as_complex'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('masked_select'),
        xfail('narrow'),  # Batching rule not implemented for `narrow.Tensor` (and view op)
        skip('nn.functional.fractional_max_pool3d'),  # generator works on cpu, fails on cuda
        xfail('__rpow__'),  # https://github.com/pytorch/functorch/issues/617
        skip('nn.functional.fractional_max_pool2d'),  # generator works on cpu, fails on cuda
        xfail('column_stack', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('svd_lowrank', ''),
        xfail('pca_lowrank', ''),
        xfail('clamp'),
        xfail('cross'),  # The defaults of this op are *very* weird. No wonder it doesn't work
        # something weird happening with channels_last
        xfail('bfloat16'),
        xfail('double'),
        xfail('float'),
        xfail('half'),
        xfail('nn.functional.dropout3d', ''),
        xfail('as_strided_scatter', ''),
        xfail('sparse.sampled_addmm', ''),
    }))
    def test_vjpvmap(self, device, dtype, op):
        # NB: there is no vjpvmap_has_batch_rule test because that is almost
        # certainly redundant with the vmap_has_batch_rule test in test_vmap.py

        # one-off skip
        if op.name == 'nn.functional.dropout':
            self.skipTest("Skipped!")

        if not op.supports_autograd:
            # If the op doesn't support autograd, vmap(op) won't either
            self.skipTest("Skipped! Autograd not supported.")
            return

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        batch_norm_fns = ("nn.functional.batch_norm", "nn.functional.instance_norm")  # instance norm calls batch norm
        is_batch_norm = op.name in batch_norm_fns

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            if is_batch_norm and is_batch_norm_training(op.name, kwargs):
                generator = get_exhaustive_batched_inputs_batch_norm_is_training(args, kwargs)
            else:
                generator = get_exhaustive_batched_inputs(args, kwargs)

            for batched_args, in_dims, kwargs in generator:
                vmapped_op = vmap(op, in_dims)
                fn, primals = normalize_op_input_output2(vmapped_op, batched_args, kwargs,
                                                         sample.output_process_fn_grad)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                _, vjp_fn = vjp(fn, *primals)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

    def _compare_jacobians_of_vjp(self, fn, cotangents_and_primals, argnums=None, atol_rtol=None):
        if argnums is None:
            argnums = tuple(range(len(cotangents_and_primals)))

        def get_vjp(cotangents, *primals):
            _, vjp_fn = vjp(fn, *primals)
            return vjp_fn(cotangents)

        jacobian_jvp = jacfwd(get_vjp, argnums)(*cotangents_and_primals)
        jacobian_vjp = jacrev(get_vjp, argnums)(*cotangents_and_primals)

        # For dtype changing operations, the jacobians have different dtype.
        jacobian_jvp = tree_map(lambda x: x.to(torch.float), jacobian_jvp)
        jacobian_vjp = tree_map(lambda x: x.to(torch.float), jacobian_vjp)

        if atol_rtol is not None:
            (atol, rtol) = atol_rtol
            self.assertEqual(jacobian_jvp, jacobian_vjp, atol=atol, rtol=rtol)
        else:
            self.assertEqual(jacobian_jvp, jacobian_vjp)

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_jvpvjp', vjp_fail.union({
        xfail('to_sparse', ''),  # NYI
        # RuntimeError: Trying to set a forward gradient that has a different size than that of the original Tensor,
        # this is not supported. Tensor is of size [5, 2, 3] while the given forward gradient is of size [1, 2, 3].
        xfail('normal', ''),
        xfail('cdist', ''),  # NYI: forward-AD for _cdist_forward
        xfail('cholesky', ''),  # NYI: forward-AD for cholesky
        xfail('logcumsumexp', ''),  # NYI: forward-AD for logcumsumexp
        xfail('nn.functional.embedding_bag', ''),  # NYI: forward-AD for _embedding_bag
        xfail('nn.functional.grid_sample', ''),  # NYI: forward AD for grid_sampler_2d
        xfail('nn.functional.hardsigmoid', ''),  # NYI: forward AD for hardsigmoid_backward
        xfail('nn.functional.huber_loss', ''),  # NYI: forward AD for huber_loss_backward
        xfail('nn.functional.logsigmoid', ''),  # not differentiable w.r.t. buffer
        xfail('renorm', ''),  # NYI: forward AD for renorm
        xfail('symeig', ''),  # NYI: forward AD for symeig
        xfail('nn.functional.multilabel_margin_loss', ''),  # NYI: multilabel_margin_loss_forward
        xfail('nn.functional.multilabel_soft_margin_loss', ''),  # NYI: log_sigmoid_backward
        xfail('nn.functional.soft_margin_loss', ''),  # NYI: forward-AD for log_sigmoid_backward
        xfail('nn.functional.ctc_loss', ''),  # NYI: forward-AD for _ctc_loss
        xfail('nn.functional.pdist', ''),  # NYI: forward-AD with _pdist_forward
        xfail('nn.functional.multi_margin_loss', ''),  # NYI: forward AD with multi_margin_loss
        skip('linalg.householder_product', '', device_type='cuda'),  # flaky, I'm not sure why
        xfail('sparse.sampled_addmm', ''),  # Sparse tensors have no strides
        skip('as_strided_scatter', ''),  # seems flaky
        xfail('segment_reduce', 'offsets'),  # NYI: forward-AD for segment_reduce
        xfail('index_reduce', ''),  # NYI: forward-AD for index_reduce
        xfail('segment_reduce', 'lengths'),  # NYI: forward-AD for segment_reduce
    }))
    @opsToleranceOverride('TestOperators', 'test_jvpvjp', (
        tol1('_masked.prod',
             {torch.float32: tol(atol=1e-04, rtol=1.3e-05)}),
        tol1('_masked.cumprod',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
        tol1('cumprod',
             {torch.float32: tol(atol=1e-04, rtol=1.3e-05)}, device_type='cuda'),
        tol1('linalg.vander',
             {torch.float32: tol(atol=1e-04, rtol=1.3e-05)}, device_type='cuda'),
    ))
    def test_jvpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            result = jvp(push_vjp, (primals, cotangents), (primals_tangents, cotangents_tangents))
            self.assertEqual(len(result), 2)

            def tree_map2(fn, first, second):
                flat_first, spec_first = tree_flatten(first)
                flat_second, spec_second = tree_flatten(second)
                assert spec_first == spec_second
                flat_result = [fn(f, s) for f, s in zip(flat_first, flat_second)]
                return tree_unflatten(flat_result, spec_first)

            def reference(primals, cotangents, primals_tangents, cotangents_tangents):
                with fwAD.dual_level():
                    primal_duals = tree_map2(fwAD.make_dual, primals, primals_tangents)
                    _, vjp_fn = ref_vjp(fn, *primal_duals)

                    cotangent_duals = tree_map2(fwAD.make_dual, cotangents, cotangents_tangents)
                    result = vjp_fn(cotangent_duals)

                    flat_result, spec = tree_flatten(result)
                    primals_out, tangents_out = zip(*[fwAD.unpack_dual(r) for r in flat_result])
                    tangents_out = [t if t is not None else torch.zeros_like(p)
                                    for p, t in zip(primals_out, tangents_out)]
                    expected = (tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec))
                return expected

            expected = reference(primals, cotangents, primals_tangents, cotangents_tangents)
            self.assertEqual(result, expected)

    @skipOps('TestOperators', 'test_vmapjvpvjp', vjp_fail.union({
        # Following operatos take too long, hence skipped
        skip('atleast_1d'),
        skip('atleast_2d'),
        skip('atleast_3d'),
        skip('meshgrid', 'list_of_tensors'),
        skip('meshgrid', 'variadic_tensors'),
        skip('broadcast_tensors'),
        skip('linalg.lstsq'),
        skip('nn.functional.bilinear'),
        skip('native_layer_norm'),

        # Potential bugs/errors
        xfail('as_strided'),  # AssertionError: Tensor-likes are not close!
        xfail('as_strided_scatter'),  # AssertionError: Tensor-likes are not close!
        xfail('bernoulli'),  # calls random op
        xfail('bfloat16'),  # required rank 4 tensor to use channels_last format
        xfail('cdist'),  # Forward AD not implemented and no decomposition
        xfail('chalf'),  # required rank 4 tensor to use channels_last format
        xfail('cholesky'),  # Forward AD not implemented and no decomposition
        xfail('double'),  # required rank 4 tensor to use channels_last format
        xfail('float'),  # required rank 4 tensor to use channels_last format
        xfail('half'),  # required rank 4 tensor to use channels_last format
        xfail('index_reduce'),  # Forward AD not implemented and no decomposition
        xfail('linalg.eig'),  # vmap over torch.allclose isn't supported yet.
        # AssertionError: Tensor-likes are not close!
        # Mismatched elements: 2 / 120 (1.7%)
        # Greatest absolute difference: 0.09438323974609375
        # Greatest relative difference: 0.00115722746596277
        xfail('linalg.householder_product', device_type='cuda'),
        xfail('logcumsumexp'),  # Forward AD not implemented and no decomposition
        xfail('mvlgamma', 'mvlgamma_p_1'),  # vmap: inplace into a regular tensor
        xfail('mvlgamma', 'mvlgamma_p_3'),  # vmap: inplace into a regular tensor
        xfail('mvlgamma', 'mvlgamma_p_5'),  # vmap: inplace into a regular tensor
        xfail('nanquantile'),  # Batching rule not implemented for aten::equal
        # RuntimeError: Batch norm got a batched tensor as input while the
        # running_mean or running_var, which will be updated in place,
        # were not batched.
        xfail('nn.functional.batch_norm'),
        xfail('nn.functional.batch_norm', 'without_cudnn'),
        xfail('nn.functional.binary_cross_entropy'),  # vmap: inplace into a regular tensor
        xfail("nn.functional.ctc_loss"),  # ForwardAD not implemented and no decomposition
        xfail('nn.functional.dropout2d'),  # calls random op
        xfail('nn.functional.dropout3d'),  # calls random op
        xfail('nn.functional.dropout'),  # calls random op
        skip('nn.functional._scaled_dot_product_attention'),  # randomness
        xfail('nn.functional.embedding_bag'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),  # calls random op
        xfail('nn.functional.fractional_max_pool2d'),  # calls random op
        xfail('nn.functional.fractional_max_pool3d'),  # calls random op
        xfail('nn.functional.gaussian_nll_loss'),  # data depenedant flow
        xfail('nn.functional.grid_sample'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.hardsigmoid'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.hinge_embedding_loss'),  # vmap: inplace into a regular tensor
        xfail('nn.functional.huber_loss'),  # Forward AD not implemented and no decomposition
        # RuntimeError: Batch norm got a batched tensor as input while the
        # running_mean or running_var, which will be updated in place,
        # were not batched.
        xfail('nn.functional.instance_norm'),
        xfail('nn.functional.logsigmoid'),  # Forward AD not implemented and no decomposition
        # NYI: Tensor.clone(memory_format) inside vmap is only supported with
        # memory_format torch.preserve_format or torch.contiguous_format (got ChannelsLast)
        xfail('nn.functional.max_unpool2d'),
        xfail('nn.functional.max_unpool2d', 'grad'),
        xfail('nn.functional.multi_margin_loss'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.multilabel_margin_loss'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.multilabel_soft_margin_loss'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.pdist'),  # Forward AD not implemented and no decomposition
        xfail('nn.functional.rrelu'),  # vmap: we do not yet support aten::rrelu_with_noise.
        xfail('nn.functional.soft_margin_loss'),  # Forward AD not implemented and no decomposition
        xfail('normal'),  # calls random op
        xfail('normal', 'number_mean'),  # calls random op
        xfail('pca_lowrank'),  # calls random op
        xfail('quantile'),  # Batching rule not implemented for aten::equal
        xfail('renorm'),  # Forward AD not implemented and no decomposition
        xfail('scatter_reduce', 'prod'),  # Forward AD not implemented and no decomposition
        xfail('segment_reduce', 'lengths'),  # Forward AD not implemented and no decomposition
        xfail('segment_reduce', 'offsets'),  # Forward AD not implemented and no decomposition
        xfail('sparse.sampled_addmm'),  # RuntimeError: Sparse CSR tensors do not have strides
        xfail('svd_lowrank'),  # calls random op
        xfail('symeig'),  # Forward AD not implemented and no decomposition
        xfail('take'),  # vmap: inplace into regular tensor
        xfail('to'),  # RuntimeError: required rank 4 tensor to use channels_last format
        xfail('to_sparse'),  # Forward AD not implemented and no decomposition
        xfail('view_as_complex'),  # RuntimeError: Tensor must have a last dimension with stride 1
    }))
    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @opsToleranceOverride('TestOperators', 'test_vmapjvpvjp', (
        tol1('linalg.svd',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('linalg.householder_product',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('linalg.multi_dot',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
        tol1('svd',
             {torch.float32: tol(atol=5e-04, rtol=5e-04)}),
    ))
    def test_vmapjvpvjp(self, device, dtype, op):
        # Since we test `jvpvjp` seperately,
        # in this we just check that vmap of `jvpvjp`
        # is correct.
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            args, spec = tree_flatten(((primals, cotangents), (primals_tangents, cotangents_tangents)))

            def jvp_of_vjp(*args):
                (primals, tangents) = tree_unflatten(args, spec)
                primals_out, tangents_out = jvp(push_vjp, primals, tangents)

                flat_primals_out, _ = tree_flatten(primals_out)
                flat_tangents_out, _ = tree_flatten(tangents_out)
                return tuple(flat_primals_out + flat_tangents_out)

            is_batch_norm_and_training = is_batch_norm_training(op, sample.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                jvp_of_vjp, args, {}, is_batch_norm_and_training=is_batch_norm_and_training)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)


    def _make_extremal_inputs(self, shape, device):
        if shape is None:
            return (None,)
        return (
            torch.full(shape, -1000., device=device),
            torch.zeros(shape, device=device),
            torch.full(shape, 1000., device=device),
        )

    def _arg_and_kwarg_options(self, args_options, kwargs_options):
        return itertools.product(*args_options, kwargs_options)

    def test_extremal_numerics_nll_loss(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
        )
        kwargs_options = ({'ignore_index': 0, 'reduction': 'mean'}, {'reduction': 'sum'}, {'reduction': 'none'}, {})
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)
                target = torch.randint(0, C, target_shape, device=device)
                target[0] = 1  # since we're ignoring index 0, at least one element must be non-zero

                fn = functools.partial(torch.nn.functional.nll_loss, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input))

    def test_extremal_numerics_l1_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'reduction': 'sum'}, {'reduction': 'none'}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options((input_options, target_options), kwargs_options):
                result = torch.nn.functional.l1_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.l1_loss, (cotangents, input, target))

    def test_extremal_numerics_mse_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'reduction': 'sum'}, {'reduction': 'none'}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options((input_options, target_options), kwargs_options):
                result = torch.nn.functional.mse_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.mse_loss, (cotangents, input, target))

    def test_extremal_numerics_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'dim': 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                result = torch.nn.functional.softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.softmax, (cotangents, input))


    def test_extremal_numerics_log_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'dim': 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                result = torch.nn.functional.log_softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.log_softmax, (cotangents, input))

    def test_extremal_numerics_cross_entropy(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C), (N, C), (C,)),
            ((N, C), (N, C), None),
            ((C,), (), (C,)),
            ((C,), (), None),
            ((C,), (C,), (C,)),
            ((C,), (C,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), None),
        )
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            kwargs_options = [{'reduction': 'sum'}, {'reduction': 'none'}, {}]
            if input_shape != target_shape:
                kwargs_options.append({'ignore_index': 0, 'reduction': 'mean'})

            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)

                if input_shape == target_shape:
                    target = torch.rand(target_shape, device=device)
                elif len(target_shape) == 0:
                    target = torch.tensor(1, device=device)  # must be non-zero since ignore_index may be 0
                else:
                    target = torch.randint(0, C, target_shape, device=device)

                fn = functools.partial(torch.nn.functional.cross_entropy, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input), atol_rtol=(1e-4, 1e-5))

    def test_extremal_numerics_binary_cross_entropy(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            weight_options = self._make_extremal_inputs(shape, device)
            kwargs_options = [{'reduction': 'sum'}, {'reduction': 'none'}, {}]

            for weight, kwargs in self._arg_and_kwarg_options((weight_options,), kwargs_options):
                input = torch.rand(shape, device=device)
                target = torch.rand(shape, device=device)
                fn = functools.partial(torch.nn.functional.binary_cross_entropy, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input), atol_rtol=(1e-4, 2e-5))

    def test_extremal_numerics_layer_norm(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            normalized_shape = shape[1:]
            weight_options = self._make_extremal_inputs(normalized_shape, device)
            bias_options = self._make_extremal_inputs(normalized_shape, device)

            for input, bias, weight in self._arg_and_kwarg_options((input_options, bias_options, weight_options), ()):
                def fn(input, weight, bias):
                    return torch.nn.functional.layer_norm(input, normalized_shape, weight=weight, bias=bias)
                result = fn(input, weight, bias)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input, weight, bias))

    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float32, torch.double))
    @skipOps('TestOperators', 'test_vmap_autograd_grad', {
        xfail('linalg.eig'),  # all close?
        # The size of tensor a (4) must match the size of tensor b (10) at non-singleton dimension 0
        xfail('masked_select'),
        xfail('nn.functional.max_unpool2d', 'grad'),  # contiguous call
        xfail('nn.functional.max_unpool2d'),  # contiguous call
        xfail('to_sparse'),  # dispatch key issue

        # numerical inconsistencies, look like bugs
        skip('matrix_exp', dtypes=(torch.float32,), device_type='cuda'),  # fails on linux, passes on windows
        skip('ldexp', dtypes=(torch.float32,), device_type='cpu'),  # fails on all but mac
        skip('__rmatmul__'),  # flaky needs investigation
        skip('matmul'),  # flaky needs investigation
        skip('nn.functional.conv_transpose3d'),  # flaky needs investigation
        skip('nn.functional.conv_transpose2d'),  # flaky needs investigation
        skip('nn.functional.conv_transpose1d'),  # flaky needs investigation
        skip('nn.functional.layer_norm', dtypes=(torch.float32,), device_type='cpu'),  # fails on windows
        skip('linalg.lu_factor', dtypes=(torch.float32,), device_type='cuda'),  # fails on all but windows
        skip('linalg.lu_factor_ex', dtypes=(torch.float32,), device_type='cuda'),  # fails on all but windows
        skip('linalg.multi_dot', '', device_type='cpu'),
        skip('sparse.sampled_addmm', ''),
        skip('native_layer_norm', '', device_type='cpu'),
        xfail('as_strided_scatter', ''),
    })
    @opsToleranceOverride('TestOperators', 'test_vmap_autograd_grad', (
        tol1('linalg.householder_product',
             {torch.float32: tol(atol=5e-04, rtol=9e-03)}, device_type='cuda'),
        tol1('linalg.householder_product',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}, device_type='cpu'),
    ))
    def test_vmap_autograd_grad(self, device, dtype, op):
        def is_differentiable(inp):
            return isinstance(inp, Tensor) and (inp.grad_fn is not None or inp.requires_grad)

        def get_flat_differentiable(pytree):
            flattened = tree_flatten(pytree)[0]
            return tuple(i for i in flattened if is_differentiable(i))

        def get_differentiable_linked(list1, list2):
            paired_list = zip(list1, list2)
            paired_list = tuple((first, second) for (first, second) in paired_list if is_differentiable(first))
            return zip(*paired_list)

        def filter_none(out):
            flattened = tree_flatten(out)[0]
            return tuple(o for o in flattened if o is not None)

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs:
            fn, primals = normalize_op_input_output(op, sample_input)
            out = fn(*primals)
            cotangents = tree_map(torch.randn_like, out)

            def compute_grad(cotangents):
                out_flattened = out
                cotangents_flattened = cotangents
                if not isinstance(out_flattened, torch.Tensor):
                    out_flattened = tree_flatten(out)[0]
                    cotangents_flattened = tree_flatten(cotangents)[0]
                    out_flattened, cotangents_flattened = get_differentiable_linked(out_flattened, cotangents_flattened)

                return filter_none(
                    torch.autograd.grad(out_flattened, get_flat_differentiable(primals), cotangents_flattened,
                                        retain_graph=True, allow_unused=True))

            is_batch_norm_and_training = is_batch_norm_training(op, sample_input.kwargs)
            generator = get_fallback_and_vmap_exhaustive(
                compute_grad, (cotangents,), {}, is_batch_norm_and_training=is_batch_norm_and_training)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    def test_vmapvmapjvp_linalg_solve(self):
        ops = [op for op in op_db if op.name == "linalg.solve"]
        assert len(ops) > 0

        # this specializes a lot of code from the get_fallback_and_vmap_exhaustive test. If we need this more
        # generally, this could go for a refactor

        B0 = 2
        B1 = 3

        # we want to check the case where A will be seen as contiguous by jvp but during the vmap calls will become
        # non-contiguous because vmap will expand. This will happen during both levels of vmap
        A = torch.randn(4, 4)
        k = torch.randn(4, 5, B1, B0)
        fn, args = get_jvp_variant_primals_tangents(torch.linalg.solve, SampleInput(A, args=(k,)))

        in_dims_all = (None, -1, None, -1)
        batched_out = vmap(vmap(fn, in_dims=in_dims_all), in_dims=in_dims_all)(*args)
        loop_out = loop2(fn, in_dims_all, in_dims_all, 0, 0, B0, B1, *args)
        self.assertEqual(loop_out, batched_out)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()