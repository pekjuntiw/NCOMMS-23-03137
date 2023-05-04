import torch
from spikingjelly.clock_driven import neuron


class scaled_piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.gamma = gamma
        return neuron.surrogate.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x_abs = ctx.saved_tensors[0].abs()
            mask = (x_abs > (1 / ctx.alpha))
            grad_x = (grad_output * (- (ctx.alpha ** 2) * x_abs + ctx.alpha) * ctx.gamma).masked_fill_(mask, 0)
        return grad_x, None, None


class ScaledPiecewiseQuadratic(neuron.surrogate.MultiArgsSurrogateFunctionBase):
    def __init__(self, alpha=1.0, gamma=0.3, spiking=True):
        super().__init__(spiking)
        self.alpha = alpha
        self.gamma = gamma
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.alpha, self.gamma)

    @staticmethod
    def spiking_function(x, alpha, gamma):
        return scaled_piecewise_quadratic.apply(x, alpha, gamma)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha, gamma):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() * gamma + alpha * x * gamma + 0.5)