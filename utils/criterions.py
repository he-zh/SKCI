import torch
import torch.nn as nn
from utils.matrix_processing import add_diag


class LeaveOneOutKRR(nn.Module):

    def forward(self, K_XX, K_YY, reg):
        n = K_XX.shape[0]
        K_XX = K_XX.double()
        K_YY = K_YY.double()
        reg = reg.double()
        Kinv = torch.linalg.solve(add_diag(K_XX, n * reg), K_XX).T

        value = ((K_YY.diagonal() + (Kinv @ K_YY @ Kinv.T).diagonal() -
                2 * (Kinv @ K_YY).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()
        return value.float()
    # def forward(self, K_yy, K_QQ, reg):
    #     n = K_yy.shape[0]
    #     K_yy = K_yy.double()
    #     K_QQ = K_QQ.double()
    #     reg = reg.double()
    #     Kinv = torch.linalg.solve(add_diag(K_yy, n * reg), K_yy).T

    #     value = ((K_QQ.diagonal() + (Kinv @ K_QQ @ Kinv.T).diagonal() -
    #             2 * (Kinv @ K_QQ).diagonal()) / (1 - Kinv.diagonal()) ** 2).mean()
    #     return value.float()


class SquareLossKRR(nn.Module):
    def forward(self, K_xX, K_XX, K_yy, K_yY, K_YY, reg):
        n = K_XX.shape[0]
        K_XX = K_XX.double()
        K_YY = K_YY.double()
        K_yy = K_yy.double()
        K_xX = K_xX.double()
        K_yY = K_yY.double()
        reg = reg.double()
        Kinv = torch.linalg.solve(add_diag(K_XX, n * reg), K_xX.T).T

        value = (K_yy.diagonal() + (Kinv @ K_YY @ Kinv.T).diagonal() -
                2 * (Kinv @ K_yY.T).diagonal()).mean()
        return value.float()


def compute_hsic(Kx, Ky, biased):
    n = Kx.shape[0]
    Kx = Kx.double()
    Ky = Ky.double()
    if biased:
        a_vec = Kx.mean(dim=0)
        b_vec = Ky.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        mean =  (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()
        return mean.float()


    else:
        tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
        tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

        u = tilde_Kx * tilde_Ky
        k_row = tilde_Kx.sum(dim=1)
        l_row = tilde_Ky.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_Kx.sum()
        mu_y = tilde_Ky.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean.float()


def compute_unbiased_kci(K_aa_centered, K_bb_centered, K_cc):
    n = K_aa_centered.size(0)
    hh = K_aa_centered * K_bb_centered * K_cc
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=hh.device)
    off_diag_mean = hh[off_diag_mask].mean()
    return off_diag_mean.float()