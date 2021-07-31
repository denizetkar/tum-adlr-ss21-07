from typing import Any, Optional, Tuple, Union

import torch as th
from torch.autograd import Function


class SmartClamp(Function):
    """SmartClamp allows gradients to flow if the gradient updates are going to
    decrease those values that are bigger than the maximum or increase those values
    that are smaller than the minimum. Therefore, the gradient update direction
    should be specified using 'is_gradient_descent' argument in the 'forward' method.
    Otherwise, it will by default behave the same way as 'torch.clamp' method.

    NOTE: Please only use the gradients in either gradient ascent or gradient descent!
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: th.Tensor,
        minimum: Optional[Union[float, th.Tensor]] = None,
        maximum: Optional[Union[float, th.Tensor]] = None,
        is_gradient_descent: Optional[bool] = None,
    ) -> th.Tensor:
        bigger_than_max = x > maximum if maximum is not None else th.zeros_like(x).bool()
        smaller_than_min = x < minimum if minimum is not None else th.zeros_like(x).bool()
        ctx.save_for_backward(bigger_than_max, smaller_than_min)
        ctx.is_gradient_descent = is_gradient_descent
        max_clamped = (
            (
                x.masked_scatter(bigger_than_max, maximum)
                if isinstance(maximum, th.Tensor)
                else x.masked_fill(bigger_than_max, maximum)
            )
            if maximum is not None
            else x.detach()
        )
        min_max_clamped = (
            (
                max_clamped.masked_scatter(smaller_than_min, minimum)
                if isinstance(minimum, th.Tensor)
                else max_clamped.masked_fill(smaller_than_min, minimum)
            )
            if minimum is not None
            else max_clamped
        )
        return min_max_clamped

    @staticmethod
    def backward(ctx: Any, grad_output: th.Tensor) -> Tuple[Optional[th.Tensor], None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None

        grad_input = grad_output.clone()
        bigger_than_max: th.Tensor
        smaller_than_min: th.Tensor
        bigger_than_max, smaller_than_min = ctx.saved_tensors
        if ctx.is_gradient_descent is None:
            grad_input[th.logical_or(bigger_than_max, smaller_than_min)] = 0.0
            return grad_input, None, None, None

        grad_input[
            th.logical_or(
                th.logical_and(grad_output > 0, smaller_than_min if ctx.is_gradient_descent else bigger_than_max),
                th.logical_and(grad_output < 0, bigger_than_max if ctx.is_gradient_descent else smaller_than_min),
            )
        ] = 0.0
        return grad_input, None, None, None


def smart_clamp(
    x: th.Tensor,
    minimum: Optional[Union[float, th.Tensor]] = None,
    maximum: Optional[Union[float, th.Tensor]] = None,
    is_gradient_descent: Optional[bool] = None,
) -> th.Tensor:
    return SmartClamp.apply(x, minimum, maximum, is_gradient_descent)
