import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Callable
from spikingjelly.clock_driven import neuron, base
from surrogate import ScaledPiecewiseQuadratic
from utils import absId


class LIFVO2(neuron.BaseNode):
    """
        LIF neuron based on VO2 LIF circuit with a current input
    """
    def __init__(self,
                 Rh: float = 100e3, Rs: float = 1.5e3,
                 v_threshold: float = 2., v_reset: float = 0.5,
                 Cmem: float = 200e-9,
                 Vdd: float = 5.,
                 input_scaling: float = 250e-6,
                 dt: float = 1e-3, refractory: int = 5,
                 surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.Rh = Rh
        self.Rs = Rs
        self.Cmem = Cmem
        self.tau = Cmem * (Rh + Rs)
        self.factor = torch.exp(torch.tensor(-dt / self.tau))
        self.refractory = refractory
        self.dt = dt

        self.Vdd = Vdd

        self.input_scaling = input_scaling

        self.v = 0.
        self.register_memory('va', 0.)
        self.register_memory('spike', 0)
        self.register_memory('spike_countdown', None)
        self.register_memory('is_refractory', 0)

    def neuronal_charge(self, x: torch.Tensor):
        x = torch.relu(x)  # VO2 LIF circuit cannot accept -ve inputs, assuming input via MOSFET, -ve input = 0 current
        v = self.factor * self.v + (1 - self.factor) * (self.Rh + self.Rs) * (self.input_scaling * x)
        self.v = torch.where(self.is_refractory, self.v, v)

    def neuronal_fire(self):
        # activation function, Heaviside function and surrogate function during forward and backward pass, respectively
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)

        with torch.no_grad():
            if self.spike_countdown is None:
                shape = list(spike.shape)
                shape.append(self.refractory)
                self.spike_countdown = torch.zeros(shape, device=spike.device)

        spike = torch.where(self.is_refractory, torch.zeros_like(spike), spike)
        self.spike = spike
        self.spike_countdown = torch.cat([self.spike_countdown[:, :, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)

        return self.spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.is_refractory, torch.Tensor):
                self.is_refractory = torch.zeros(x.shape, device=x.device, dtype=torch.bool)

        self.neuronal_reset(self.spike)
        self.neuronal_charge(x)
        new_spike = self.neuronal_fire()
        self.is_refractory = torch.gt(torch.amax(self.spike_countdown[..., -self.refractory:], dim=-1), 0)

        return new_spike


class ALIFVO2(neuron.BaseNode):
    """
        ALIF neuron based on VO2 ALIF circuit with a current input
        LIF circuit with a spike feedback path to control the adaptation circuit
        Adaptation circuit dynamically controls a membrane leak path to achieve adaptation
    """
    def __init__(self,
                 Rh: float = 100e3, Rs: float = 1.5e3, Ra: float = 100e3,
                 v_threshold: float = 2., v_reset: float = 0.5,
                 Cmem: float = 200e-9, Ca: float = 7e-6,
                 vtn: float = 0.745, vtp: float = 0.973, kappa_n: float = 29e-6, kappa_p: float = 18e-6,
                 wl_ratio_n: float = 6., wl_ratio_p: float = 4.,
                 Vdd: float = 5.,
                 input_scaling: float = 250e-6,
                 dt: float = 1e-3, refractory: int = 5,
                 surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.Rh = Rh
        self.Rs = Rs
        self.Cmem = Cmem
        self.tau = Cmem * (Rh + Rs)
        self.factor = torch.exp(torch.tensor(-dt / self.tau))
        self.refractory = refractory
        self.dt = dt

        self.Ra = Ra
        self.Ca = Ca
        self.tau_va = Ca * Ra
        self.factor_va = torch.exp(torch.tensor(-dt / self.tau_va))

        self.Vdd = Vdd
        self.vtn = vtn
        self.vtp = vtp
        self.kappa_n = kappa_n
        self.kappa_p = kappa_p
        self.wl_ratio_n = wl_ratio_n
        self.wl_ratio_p = wl_ratio_p

        self.input_scaling = input_scaling

        self.v = 0.
        self.register_memory('va', 0.)
        self.register_memory('spike', 0)
        self.register_memory('spike_countdown', None)
        self.register_memory('is_refractory', 0)

    def neuronal_charge(self, x: torch.Tensor):
        x = torch.relu(x)  # VO2 LIF circuit cannot accept -ve inputs, assuming input via MOSFET, -ve input = 0 current
        Il = absId(kappa=self.kappa_n, w_over_l=self.wl_ratio_n, vgs=self.va,
                   vth=self.vtn, vds=self.v)  # leak current via MOSFET for frequency adaptation
        v = self.factor * self.v + (1 - self.factor) * (self.Rh + self.Rs) * (self.input_scaling * x - Il)
        self.v = torch.where(self.is_refractory, self.v, v)

    def neuronal_fire(self):
        # activation function, Heaviside function and surrogate function during forward and backward pass, respectively
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)

        with torch.no_grad():
            if self.spike_countdown is None:
                shape = list(spike.shape)
                shape.append(self.refractory)
                self.spike_countdown = torch.zeros(shape, device=spike.device)

        spike = torch.where(self.is_refractory, torch.zeros_like(spike), spike)
        self.spike = spike
        self.spike_countdown = torch.cat([self.spike_countdown[:, :, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)

        return self.spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def neuronal_adapt(self, spike):
        Ia = absId(kappa=self.kappa_p, w_over_l=self.wl_ratio_p, vgs=(self.Vdd * spike),
                   vth=self.vtp, vds=(self.Vdd - self.va))  # charge adaptation circuit
        self.va = self.factor_va * self.va + (1 - self.factor_va) * self.Ra * Ia

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.va, torch.Tensor):
                self.va = torch.zeros(x.shape, device=x.device)
            if not isinstance(self.is_refractory, torch.Tensor):
                self.is_refractory = torch.zeros(x.shape, device=x.device, dtype=torch.bool)

        self.neuronal_reset(self.spike)
        self.neuronal_adapt(self.spike)
        self.neuronal_charge(x)
        new_spike = self.neuronal_fire()
        self.is_refractory = torch.gt(torch.amax(self.spike_countdown[..., -self.refractory:], dim=-1), 0)

        return new_spike


class LPFilter(base.MemoryModule):
    """
        Low-pass filter, think of it as a leaky spike counter
        Modeled using an RC circuit
    """
    def __init__(self, tau: float = 20e-3, dt: float = 1e-3):
        super(LPFilter, self).__init__()
        self.tau = tau
        self.dt = dt
        self.factor = torch.exp(torch.tensor(-dt / tau))
        self.register_memory("out", 0)

    def forward(self, x: torch.Tensor):
        self.out = self.factor * self.out + (1 - self.factor) * x

        return self.out


class DelayedLinear(nn.Linear):
    """
        Linear fully-connected layer with randomly initialized synaptic delay
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_delay: int = 10,
                 diag_disconnect: bool = False):
        super(DelayedLinear, self).__init__(in_features, out_features, bias)
        if diag_disconnect:
            assert in_features == out_features, "in_features == out_features must be True for diagonal disconnecting feature"

        self.max_delay = max_delay
        self.delays = torch.randint(max_delay, size=(out_features, in_features))
        self.diag_disconnect = diag_disconnect
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("delay_mask", torch.Tensor())

        with torch.no_grad():
            self.weight = torch.nn.parameter.Parameter(torch.Tensor(out_features, in_features))

        self.custom_reset_parameters(1., 1.)
        self.set_delayed_weight()

        self.out_buf = 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delay_masked_weight = torch.transpose(torch.where(self.delay_mask, self.weight, torch.zeros_like(self.weight)),
                                              dim0=0, dim1=1)

        # implementation 1: einsum bi,ijk->bjk
        self.out_buf += self.einsum_bi_ijk_bjk(x, delay_masked_weight)

        # implementation 2: torch einsum bi,ijk->bjk, slower than 1
        # self.out_buf += torch.einsum("bi,ijk->bjk", x, delay_masked_weight)

        # implementation 3: one-liner einsum bi,ijk->bjk, same as 1
        # self.out_buf += torch.reshape(torch.mm(x, delay_masked_weight.flatten(1)), [x.shape[0], delay_masked_weight.shape[1], delay_masked_weight.shape[2]])

        out = self.out_buf[:, :, 0]
        self.roll()

        return out

    def einsum_bi_ijk_bjk(self, x: torch.Tensor, w: torch.Tensor):
        w_shape = w.shape  # (in_features, out_features, max_delay)
        x_shape = x.shape  # (batch_size, in_features)
        w_ = torch.reshape(w, (w_shape[0], (w_shape[1] * w_shape[2])))
        out_ = torch.mm(x, w_)
        out = torch.reshape(out_, (x_shape[0], w_shape[1], w_shape[2]))

        return out

    def roll(self):
        zeros = torch.zeros_like(self.out_buf[:, :, 0])
        zeros = torch.unsqueeze(zeros, dim=-1)
        self.out_buf = torch.cat([self.out_buf[:, :, 1:], zeros], dim=-1)

    def custom_reset_parameters(self, dt, R) -> None:
        # X~N(u,sigma^2)
        # then, kX~N(k*u,(k*sigma)^2)
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        scale = 1 * dt / R
        scale /= math.sqrt(fan_in)
        init.normal_(self.weight, mean=0, std=scale)

        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def set_delayed_weight(self):
        with torch.no_grad():
            delay_mask_list = []
            delayed_weight_list = []

            if self.diag_disconnect:
                diag_disconnect_mask = torch.eye(self.in_features, dtype=torch.bool, device=self.weight.device)
                self.weight = torch.nn.parameter.Parameter(
                    torch.where(diag_disconnect_mask, torch.zeros_like(self.weight), self.weight)
                )

            for delay in range(self.max_delay):
                mask = torch.eq(self.delays, delay)
                delay_mask_list.append(mask)
                delayed_weight = torch.where(mask, self.weight, torch.zeros_like(self.weight))
                delayed_weight_list.append(delayed_weight)

            delay_axis = len(self.delays.shape)
            self.delay_mask = torch.stack(delay_mask_list, dim=delay_axis)
            self.weight = torch.nn.parameter.Parameter(torch.stack(delayed_weight_list, dim=delay_axis))

    def reset(self):
        self.out_buf = 0.


class LinearDualRecurrentContainer(base.MemoryModule):
    """
        Recurrent layer with two spiking neuron models (LIF and ALIF)
    """
    def __init__(self, sub_module_1: nn.Module, sub_module_2: nn.Module,
                 in_features: int, out_features_1: int, out_features_2: int,
                 bias: bool = True, max_delay: int = 10) -> None:
        super().__init__()
        self.out_features_1 = out_features_1
        self.out_features_2 = out_features_2
        self.out_features_total = out_features_1 + out_features_2
        self.rc = DelayedLinear(self.out_features_total, in_features, bias, max_delay, diag_disconnect=True)
        self.sub_module_1 = sub_module_1
        self.sub_module_2 = sub_module_2
        self.register_memory('y', None)

    def forward(self, x: torch.Tensor):
        """
        :param x: shape [batch, *, in_features]
        :return: shape [batch, *, out_features]
        """
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.out_features_total]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.out_features_total)
                self.y = torch.zeros(out_shape).to(x)

        # combine weighted input and weighted recurrent input
        lin_rc = self.rc(self.y)
        lin = x + lin_rc

        # switch for mixed, LIF-only and ALIF-only
        if self.out_features_2 == 0:
            self.y = self.sub_module_1(lin)
        elif self.out_features_1 == 0:
            self.y = self.sub_module_2(lin)
        else:
            lin_1 = lin[..., :self.out_features_1]
            lin_2 = lin[..., self.out_features_1:self.out_features_total]
            y_1 = self.sub_module_1(lin_1)
            y_2 = self.sub_module_2(lin_2)
            self.y = torch.cat((y_1, y_2), dim=-1)

        return self.y


class VO2LSNN(nn.Module):
    """
        VO2-based LSNN using the above network modules
        Input -> Recurrent VO2 LIFs and ALIFs -> Low-pass -> Output

        Reference:
        - Bellec, G., Salaj, D., Subramoney, A., Legenstein, R. & Maass, W. Long short-term memory and learning-to-learn
          in networks of spiking neurons. Adv. Neural Inf. Process. Syst. 31 (2018).
    """
    def __init__(self, num_in, num_lif, num_alif, num_out,
                 tau, tau_lp,
                 Rh, Rs, Ra, Cmem, Ca,
                 v_threshold, v_reset,
                 vtn, vtp, kappa_n, kappa_p, wl_ratio_n, wl_ratio_p,
                 Vdd,
                 input_scaling,
                 dt, max_delay, refractory, device):
        super(VO2LSNN, self).__init__()
        self.fc1 = DelayedLinear(num_in, num_lif + num_alif, bias=False, max_delay=max_delay)
        self.hidden = LinearDualRecurrentContainer(
            sub_module_1=LIFVO2(
                Rh=Rh, Rs=Rs, Cmem=Cmem,
                v_threshold=v_threshold, v_reset=v_reset,
                Vdd=Vdd,
                input_scaling=input_scaling,
                dt=dt, refractory=refractory, surrogate_function=ScaledPiecewiseQuadratic()
            ),
            sub_module_2=ALIFVO2(
                Rh=Rh, Rs=Rs, Ra=Ra, Cmem=Cmem, Ca=Ca,
                v_threshold=v_threshold, v_reset=v_reset,
                vtn=vtn, vtp=vtp, kappa_n=kappa_n, kappa_p=kappa_p,
                wl_ratio_n=wl_ratio_n, wl_ratio_p=wl_ratio_p,
                Vdd=Vdd,
                input_scaling=input_scaling,
                dt=dt, refractory=refractory, surrogate_function=ScaledPiecewiseQuadratic()
            ),
            in_features=(num_lif + num_alif), out_features_1=num_lif, out_features_2=num_alif, bias=False,
            max_delay=max_delay
        )
        self.lp = LPFilter(tau=tau_lp, dt=dt)
        self.fc2 = nn.Linear(num_lif + num_alif, num_out, bias=False)

        self.v = None
        self.v_threshold = None
        self.spike = None
        self.spike_for_reg = None

        self.num_lif = num_lif
        self.num_alif = num_alif
        self.dt = dt
        self.device = device

    def spike_regularization(self, target_f=10, lambda_f=1e-7):
        average = torch.mean(self.spike_for_reg, dim=(0, 1)) / self.dt

        return torch.sum(torch.square(average - target_f)) * lambda_f

    def forward(self, x: torch.Tensor):
        # permute or not, ~2.2s/it
        # x.shape = [batch, times, features]
        self.spike_for_reg = torch.empty([x.shape[0], x.shape[1], (self.num_lif + self.num_alif)], device=self.device)

        if x.shape[0] == 1:
            self.v = torch.empty([x.shape[1], (self.num_lif + self.num_alif)])
            self.v_threshold = torch.empty([x.shape[1], self.num_alif])
            self.spike = torch.empty([x.shape[1], (self.num_lif + self.num_alif)])

        y_seq = []

        x = x.permute(1, 0, 2)  # [times, batch, features]

        for t in range(x.shape[0]):
            y = self.fc1(x[t])
            y = self.hidden(y)
            self.spike_for_reg[:, t] = y
            y = self.lp(y)
            y_seq.append(self.fc2(y).unsqueeze(0))

            if x.shape[1] == 1:
                if self.num_alif == 0:
                    self.v[t] = self.hidden.sub_module_1.v
                    self.v_threshold[t] = self.hidden.sub_module_2.va  # no use
                    self.spike[t] = self.hidden.sub_module_1.spike
                elif self.num_lif == 0:
                    self.v[t] = self.hidden.sub_module_2.v
                    self.v_threshold[t] = self.hidden.sub_module_2.va
                    self.spike[t] = self.hidden.sub_module_2.spike
                else:
                    self.v[t] = torch.cat([self.hidden.sub_module_1.v, self.hidden.sub_module_2.v], -1)
                    self.v_threshold[t] = self.hidden.sub_module_2.va
                    self.spike[t] = torch.cat([self.hidden.sub_module_1.spike, self.hidden.sub_module_2.spike], -1)

        return torch.cat(y_seq, 0)
