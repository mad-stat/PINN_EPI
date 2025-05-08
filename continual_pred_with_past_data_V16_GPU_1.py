# Version 16 of continual prediction with all past data


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error as mae

import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.autograd import grad

import models_V5 as models


GPU_DEVICE = "cuda:1"


parser = ArgumentParser()

parser.add_argument("--device", type=str)
parser.add_argument("--resfile", type=str)
parser.add_argument("--weekmax", type=int, default=133)
parser.add_argument("--trainfrom", type=int, default=20)
parser.add_argument("--predfrom", type=int, default=40)
parser.add_argument("--l2", type=float, default=0)
parser.add_argument("--iter", type=int, default=50000)
parser.add_argument("--wode", type=float, default=1.0)
parser.add_argument("--wode_dr", type=float, default=1.0)
parser.add_argument("--winit", type=float, default=1.0)
parser.add_argument("--wZr", type=float, default=1.0)
parser.add_argument("--wDr", type=float, default=1.0)
parser.add_argument("--wA", type=float, default=1.0)
parser.add_argument("--wres", type=float, default=1.0)
parser.add_argument("--wdoses", type=float, default=1.0)
parser.add_argument("--wtrans", type=float, default=1.0)
arguments = parser.parse_args()
print(arguments)


############################# load data #############################
data = sio.loadmat("data/data_weekly_avg.mat")

res = data["res"].T
doses = data["doses"].T
Zr = data["cases"].T
Dr = data["deaths"].T
A = data["hospitalized"].T
t = data["time"].T
t_ode_all = np.arange(0, arguments.weekmax, 0.25).reshape(-1, 1)

Zr[Zr <= 0] = 0
Dr[Dr <= 0] = 0
A[A <= 0] = 0
res[res <= 0] = 0
doses[doses <= 0] = 0
doses[np.isnan(doses)] = 0


# scaling for cases, deaths, hospital admissions, residential, doses
s = [120000, 500, 2000, 20, 1e8]
Zr = Zr / s[0]
Dr = Dr / s[1]
A = A / s[2]
res = res / s[3]
doses = doses / s[4]

N = 39512223.0
eta = 1 / 4
gamma = 1 / 4
gamma_dw = 1 / 10
gamma_zw = 1
gamma_h = 1 / 10
rho = 0.5


############################# prepare data #############################
t = torch.tensor(t, dtype=torch.float32)
t_ode_all = torch.tensor(t_ode_all, dtype=torch.float32, requires_grad=True)
t0 = torch.tensor([[0]], dtype=torch.float32)

Zr0 = torch.tensor([[0]], dtype=torch.float32)
Dr0 = torch.tensor([[0]], dtype=torch.float32)
A0 = torch.tensor([[0]], dtype=torch.float32)
Zr = torch.tensor(Zr, dtype=torch.float32)
Dr = torch.tensor(Dr, dtype=torch.float32)
A = torch.tensor(A, dtype=torch.float32)
res = torch.tensor(res, dtype=torch.float32)
doses = torch.tensor(doses, dtype=torch.float32)

if arguments.device == "gpu":
    t = t.to(GPU_DEVICE)
    t_ode_all = t_ode_all.to(GPU_DEVICE)
    t0 = t0.to(GPU_DEVICE)
    Zr0 = Zr0.to(GPU_DEVICE)
    Dr0 = Dr0.to(GPU_DEVICE)
    A0 = A0.to(GPU_DEVICE)
    Zr = Zr.to(GPU_DEVICE)
    Dr = Dr.to(GPU_DEVICE)
    A = A.to(GPU_DEVICE)
    res = res.to(GPU_DEVICE)
    doses = doses.to(GPU_DEVICE)


############################# training #############################
# train_start~train_end weeks' data for training
# test_start~test_end weeks' data for testing
train_start = arguments.trainfrom  # first 20 weeks of A are nan

# prediction results
Zr_pred = [0] * arguments.weekmax
Dr_pred = [0] * arguments.weekmax
A_pred = [0] * arguments.weekmax
beta_pred = [0] * arguments.weekmax

all_Zr_pred = [[0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax]
all_Dr_pred = [[0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax]
all_A_pred = [[0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax, [0] * arguments.weekmax]


for train_end in range(arguments.predfrom, arguments.weekmax - 4 + 1, 1):
    test_start = train_end
    test_end = test_start + 4
    print("************************************************************")
    print("train data range: %d - %d" % (train_start, train_end))
    print("test data range: %d - %d" % (test_start, test_end))

    t_Zr = t[train_start:train_end]
    t_Dr = t[train_start:train_end]
    t_A = t[train_start:train_end]
    Zr_train = Zr[train_start:train_end]
    Dr_train = Dr[train_start:train_end]
    A_train = A[train_start:train_end]
    print("training data: ")
    print(Zr_train.reshape(-1))
    print(Dr_train.reshape(-1))
    print(A_train.reshape(-1))

    t_ode = t_ode_all[train_start * 4: test_end * 4 + 1]

    t_beta = t[train_start:train_end]
    res_train = res[train_start:train_end]
    doses_train = doses[train_start:train_end]
    print(res_train.reshape(-1))
    print(doses_train.reshape(-1))

    t_test = t[test_start:test_end]

    ############################# prepare model #############################
    u = models.NN(
        layers=[1, 50, 50, 50, 9],
        activation=torch.tanh,
        input_transform=lambda t: t / arguments.weekmax,
        output_transform=torch.exp,
    )
    u1 = models.NN(
        layers=[1, 50, 50, 50, 3],
        activation=torch.tanh,
        input_transform=lambda t: t / arguments.weekmax,
        output_transform=None, #torch.exp
    )

    L0 = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    logit_p_h = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    logit_p_d = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

    # parameters = list(u.parameters()) + [L0, log_beta, logit_p_h, logit_p_d]
    parameters = (
        list(u.parameters()) + list(u1.parameters()) + [L0, logit_p_h, logit_p_d]
    )
    # opt = torch.optim.Adam(parameters, lr=0.001)
    opt = torch.optim.Adam(parameters, lr=0.001, weight_decay=arguments.l2)

    weights = torch.tensor(
        [arguments.wode, arguments.winit, arguments.wZr, arguments.wDr, arguments.wA],
        dtype=torch.float32,
    )

    if arguments.device == "gpu":
        u.to(GPU_DEVICE)
        u1.to(GPU_DEVICE)
        L0 = L0.to(GPU_DEVICE)
        logit_p_h = logit_p_h.to(GPU_DEVICE)
        logit_p_d = logit_p_d.to(GPU_DEVICE)
        weights = weights.to(GPU_DEVICE)

    loss_total = []
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    loss6 = []
    loss7 = []
    loss8 = []
    min_loss = 1e10

    for it in range(arguments.iter):
        u1_pred = u1.forward(t_beta)
        u1s = torch.split(u1_pred, 3 * [1], dim=-1)
        loss_res = torch.mean((torch.exp(u1s[0]) - res_train) ** 2)
        loss_doses = torch.mean((torch.exp(u1s[1]) - doses_train) ** 2)

        p_h = torch.sigmoid(logit_p_h)
        p_d = torch.sigmoid(logit_p_d)
        args = [
            p_h,
            N,
            eta,
            gamma,
            gamma_dw,
            gamma_zw,
            gamma_h,
            p_d,
            rho,
        ]
        Y0 = eta * L0 / gamma
        Z0 = (p_h + rho * (1 - p_h)) * gamma * Y0 / gamma_zw
        H0 = p_h * gamma * Y0 / gamma_h
        D0 = p_d * gamma_h * H0 / gamma_dw
        X0 = torch.clip(N - L0 - Y0 - H0 - D0, min=N / 10)
        u0 = torch.concat([X0, L0, Y0, Z0, Zr0, H0, A0, D0, Dr0], dim=-1)

        losses = models.loss_function(
            u, t_ode, t0, u0, t_Zr, Zr_train, t_Dr, Dr_train, t_A, A_train, args, s, u1, arguments.wode_dr
        )
        loss_value = (
            torch.sum(weights * torch.stack(losses))
            + arguments.wres * loss_res
            + arguments.wdoses * loss_doses
        )
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        loss_total += [loss_value.detach().cpu().numpy()]
        # loss_ode, loss_init, loss_Zr, loss_Dr, loss_A, loss_res, loss_doses
        loss1 += [losses[0].detach().cpu().numpy()]
        loss2 += [losses[1].detach().cpu().numpy()]
        loss3 += [losses[2].detach().cpu().numpy()]
        loss4 += [losses[3].detach().cpu().numpy()]
        loss5 += [losses[4].detach().cpu().numpy()]
        loss6 += [loss_res.detach().cpu().numpy()]
        loss7 += [loss_doses.detach().cpu().numpy()]
        if it % 1000 == 0:
            print(
                it,
                loss_total[-1],
                loss1[-1],
                loss2[-1],
                loss3[-1],
                loss4[-1],
                loss5[-1],
                loss6[-1],
                loss7[-1],
            )
            if loss_total[-1] < min_loss:
                min_loss = loss_total[-1]
                torch.save(
                    u.state_dict(),
                    "./checkpoints/continual_pred_with_past_data_V16_1_model_weights",
                )
                torch.save(
                    u1.state_dict(),
                    "./checkpoints/continual_pred_with_past_data_V16_1_model_weights_1",
                )

    # make predictions
    u.load_state_dict(
        torch.load("./checkpoints/continual_pred_with_past_data_V16_1_model_weights")
    )
    u_pred = u.forward(t_test).detach().cpu()
    print("u_pred: ")
    print(u_pred[:, 4:5].reshape(-1))
    print(u_pred[:, 8:9].reshape(-1))
    print(u_pred[:, 6:7].reshape(-1))

    # for w in range(test_start, test_end, 1):
    #     Zr_pred[w] += (
    #         u_pred[w - test_start : w - test_start + 1, 4:5].item()
    #         * weights_for_pred[pred_cnt[w]]
    #     )
    #     Dr_pred[w] += (
    #         u_pred[w - test_start : w - test_start + 1, 8:9].item()
    #         * weights_for_pred[pred_cnt[w]]
    #     )
    #     A_pred[w] += (
    #         u_pred[w - test_start : w - test_start + 1, 6:7].item()
    #         * weights_for_pred[pred_cnt[w]]
    #     )
    #     beta_pred[w] += (
    #         u1_pred[w - test_start : w - test_start + 1, 2:3].item()
    #         * weights_for_pred[pred_cnt[w]]
    #     )
    #     pred_cnt[w] += 1
    
    # print("Zr_pred:")
    # print(Zr_pred)
    # print("Dr_pred:")
    # print(Dr_pred)
    # print("A_pred:")
    # print(A_pred)
    # print("beta_pred:")
    # print(beta_pred)

    for i in range(4):
        all_Zr_pred[i][test_start + i] = u_pred[i, 4:5].item()
        all_Dr_pred[i][test_start + i] = u_pred[i, 8:9].item()
        all_A_pred[i][test_start + i] = u_pred[i, 6:7].item()

    print("all_Zr_pred:")
    print(all_Zr_pred)
    print("all_Dr_pred:")
    print(all_Dr_pred)
    print("all_A_pred:")
    print(all_A_pred)
    

Zr_ref = Zr.tolist()
Dr_ref = Dr.tolist()
A_ref = A.tolist()


############################# deal with results #############################
# mae_cases = mae(
#     Zr_ref[arguments.predfrom : arguments.weekmax],
#     Zr_pred[arguments.predfrom : arguments.weekmax],
# )
# print("\nmae of cases: ", mae_cases)
# Zr_naive = Zr_ref[arguments.predfrom - 4 : arguments.weekmax - 4]
# mae_cases_naive = mae(Zr_ref[arguments.predfrom : arguments.weekmax], Zr_naive)
# print("mae of naive cases: ", mae_cases_naive)
# mase_cases = mae_cases / mae_cases_naive
# print("mase of cases: ", mase_cases)

# mae_deaths = mae(
#     Dr_ref[arguments.predfrom : arguments.weekmax],
#     Dr_pred[arguments.predfrom : arguments.weekmax],
# )
# print("\nmae of deaths: ", mae_deaths)
# Dr_naive = Dr_ref[arguments.predfrom - 4 : arguments.weekmax - 4]
# mae_deaths_naive = mae(Dr_ref[arguments.predfrom : arguments.weekmax], Dr_naive)
# print("mae of naive deaths: ", mae_deaths_naive)
# mase_deaths = mae_deaths / mae_deaths_naive
# print("mase of deaths: ", mase_deaths)

# mae_hospitalized = mae(
#     A_ref[arguments.predfrom : arguments.weekmax],
#     A_pred[arguments.predfrom : arguments.weekmax],
# )
# print("\nmae of hospitalized: ", mae_hospitalized)
# A_naive = A_ref[arguments.predfrom - 4 : arguments.weekmax - 4]
# mae_hospitalized_naive = mae(A_ref[arguments.predfrom : arguments.weekmax], A_naive)
# print("mae of naive hospitalized: ", mae_hospitalized_naive)
# mase_hospitalized = mae_hospitalized / mae_hospitalized_naive
# print("mase of hospitalized: ", mase_hospitalized)

res_dic = {}
res_dic["cases_ref"] = Zr_ref[arguments.predfrom : arguments.weekmax]
# res_dic["cases_pred"] = Zr_pred[arguments.predfrom : arguments.weekmax]
# res_dic["cases_naive"] = Zr_naive
res_dic["deaths_ref"] = Dr_ref[arguments.predfrom : arguments.weekmax]
# res_dic["deaths_pred"] = Dr_pred[arguments.predfrom : arguments.weekmax]
# res_dic["deaths_naive"] = Dr_naive
res_dic["hospitalized_ref"] = A_ref[arguments.predfrom : arguments.weekmax]
# res_dic["hospitalized_pred"] = A_pred[arguments.predfrom : arguments.weekmax]
# res_dic["hospitalized_naive"] = A_naive
# res_dic["beta_pred"] = beta_pred[arguments.predfrom : arguments.weekmax]
res_dic["all_cases_pred"] = all_Zr_pred
res_dic["all_deaths_pred"] = all_Dr_pred
res_dic["all_hospitalized_pred"] = all_A_pred
sio.savemat(arguments.resfile, res_dic)
