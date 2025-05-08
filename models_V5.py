# add weight for dr equation
import torch
import torch.nn as nn
from torch.autograd import grad


class NN(nn.Module):
    """Constructs a neural network model."""

    def __init__(
        self,
        layers,
        activation=torch.tanh,
        input_transform=None,
        output_transform=None,
    ):
        super().__init__()
        self.nn = []
        for i in range(len(layers) - 1):
            self.nn += [nn.Linear(layers[i], layers[i + 1])]

        self.nn = nn.Sequential(*self.nn)

        self.activation = activation
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, inputs):
        if self.input_transform is not None:
            outputs = self.input_transform(inputs)
        else:
            outputs = inputs
        for i in range(len(self.nn) - 1):
            outputs = self.nn[i].forward(outputs)
            outputs = self.activation(outputs)
        outputs = self.nn[-1].forward(outputs)
        if self.output_transform is not None:
            outputs = self.output_transform(outputs)
        return outputs


def lhs(t, us):
    """Computes the left hand side (LHS) of the ODE system."""
    # us = torch.split(u, 9*[1], dim=-1)
    us_t = []
    for u in us:
        us_t += [grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]]
    # return us_t
    return torch.concat(us_t, dim=-1)


def rhs(us, p_h, N, eta, gamma, gamma_dw, gamma_zw, gamma_h, p_d, rho, s, beta):
    """Computes the right hand side (RHS) of the ODE system."""
    # us = torch.split(u, 9*[1], dim=-1)
    X, L, Y, Z, Zr, H, A, D, Dr = us
    s_Zr = s[0]
    s_Dr = s[1]
    s_A = s[2]
    eqs = [
        -beta * X * Y / N,
        beta * X * Y / N - eta * L,
        eta * L - gamma * Y,
        (p_h + rho * (1 - p_h)) * gamma * Y - gamma_zw * Z,
        gamma_zw * Z / s_Zr,
        p_h * gamma * Y - gamma_h * H,
        p_h * gamma * Y / s_A,
        p_d * gamma_h * H - gamma_dw * D,
        gamma_dw * D / s_Dr,
    ]
    # return eqs
    return torch.concat(eqs, dim=-1)


def loss_function(u_fn, t_ode, t0, u0, t_Zr, Zr, t_Dr, Dr, t_A, A, args, s, u1_fn, wode_dr):
    """
    Compute the loss function, which, generally speaking, consists of two parts, data loss
    and ODE loss.

        Args:
            u_fn (torch module): The neural network model for the solution to the ODE system.
            t_ode (tensor): Time points with shape [batch_size, 1], for computing ODE loss.
            t0 (tensor): Time points for computing ODE loss, with shape [1, 1].
            u_0 (tensor): Observations of the initial condition, for computing the loss for
                initial condition, with shape [9, 1].
            t_Zr (tensor): Time points with shape [batch_size_2, 1], for computing data loss
                for Zr.
            Zr (tensor): Observations of Zr with shape [batch_size_2, 1], for computing data
                loss for Zr.
            t_Dr (tensor): Time points with shape [batch_size_3, 1], for computing data loss
                for Dr.
            Dr (tensor): Observations of Dr with shape [batch_size_3, 1], for computing data
                loss for Dr.
            t_A (tensor): Time points with shape [batch_size_4, 1], for computing data loss
                for A.
            A (tensor): Observations of A with shape [batch_size_4, 1], for computing data
                loss for A.
            args (list): A list contains all the quantities of the system. The order is as
                follows: beta, p_h, N, eta, gamma, gamma_dw, gamma_zw, gamma_h, p_d, rho.
        Returns:
            loss_value (tensor): The loss value.
    """
    # compute ODE loss
    u = u_fn.forward(t_ode)
    us = torch.split(u, 9 * [1], dim=-1)
    u1 = u1_fn.forward(t_ode)
    beta = torch.split(u1, 3 * [1], dim=-1)[2]
    eqs = lhs(t_ode, us) - rhs(*([us] + args), s, beta)
    eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9 = torch.split(eqs, 9 * [1], dim=-1)
    loss_ode = torch.mean(eq1 ** 2) + torch.mean(eq2 ** 2) + \
               torch.mean(eq3 ** 2) + torch.mean(eq4 ** 2) + \
               torch.mean(eq5 ** 2) + torch.mean(eq6 ** 2) + \
               torch.mean(eq7 ** 2) + torch.mean(eq8 ** 2) + \
               wode_dr * torch.mean(eq9 ** 2)
    # loss_ode = torch.mean((lhs(t_ode, us) - rhs(*([us] + args), s, beta)) ** 2)
    # compute initial condition loss
    u0_pred = u_fn.forward(t0)
    loss_init = torch.mean((u0_pred - u0) ** 2)
    # compute data loss for Zr
    Zr_pred = u_fn.forward(t_Zr)[:, 4:5]
    loss_Zr = torch.mean((Zr_pred - Zr) ** 2)
    # compute data loss for Dr
    Dr_pred = u_fn.forward(t_Dr)[:, 8:9]
    loss_Dr = torch.mean((Dr_pred - Dr) ** 2)
    # compute data loss for A
    A_pred = u_fn.forward(t_A)[:, 6:7]
    loss_A = torch.mean((A_pred - A) ** 2)

    losses = [loss_ode, loss_init, loss_Zr, loss_Dr, loss_A, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9]

    return losses
