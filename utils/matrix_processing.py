import torch
import numpy as np

def compute_pdist_sq(x, y):
    """compute the squared paired distance between x and y."""
    if len(x.shape) == 1:
        return (x[:, None] - y[None, :]) ** 2

    if len(x.shape) != 2:
        raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')
    if y is not None:
        if len(y.shape) != 2:
            raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')

        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
        y_norm = torch.linalg.norm(y, dim=1, keepdim=False)[None, :]

        return torch.clamp(x_norm ** 2 + y_norm ** 2 - 2.0 * x @ y.T, min=0)

    a = x.reshape(x.shape[0], -1)
    aTa = a @ a.T
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)


def add_diag(x, val):
    """Add a scalar value to the diagonal of a square matrix."""
    
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f'x is not a square matrix: shape {x.shape}')

    idx = range(x.shape[0])
    y = x.clone()
    y[idx, idx] += val
    return y


def get_centered_kernel_matrix(x, y, model_x, model_y):
    """
    Get the kernel matrix of y centered on the estimted conditional mean of y given x.
    K_yy_centered = <k(y,)-mu(k(y,)|x), k(y',)-mu(k(y',)|x')>
                    = K_yy + M + M.T
    M = (1/2 K_xX @ K_XX_inv_K_YY - K_yY) @ K_XX_inv @ K_xX.T
    Capital X denotes training data, lowercase x denotes test data. And similarly for Y/y.
    Args:
        x (torch.Tensor): Input data of shape (n, d_x).
        y (torch.Tensor): Output data of shape (n, d_y).
        model_x: Kernel model for x with attributes kernel_matrix and ridge_lambda.
        model_y: Kernel model for y with attribute kernel_matrix.
    Returns:
        K_yy_centered (torch.Tensor): Centered kernel matrix of shape (n, n).
    """

    # Compute the kernel matrices for y
    K_yy = model_y(y, y).double()
    # Compute the inverse kernel matrices for X
    K_XX_inv, K_XX_inv_K_YY = solve_regularized_kernel_matrix_system(model_x.kernel_matrix, model_y.kernel_matrix, 
                                                        model_x.ridge_lambda)
    # Compute the matrix M
    K_xX, K_yY = model_x(x).double(), model_y(y).double()
    M = (0.5 * K_xX @ K_XX_inv_K_YY - K_yY) @ K_XX_inv @ K_xX.T

    # Compute the centered kernel matrix for y
    K_yy_centered = K_yy + M + M.T

    return K_yy_centered.detach().float()


def get_centered_cross_kernel_matrix(x1, y1, x2, y2, model_x, model_y):
    """
    Get the kernel matrix of y centered on the estimted conditional mean of y given x.
    The input size is n1 for y1/x1 and n2 for y2/x2, and each entry is computed as
    K_y1y2_centered = <k(y1,)-mu(k(y,)|x1), k(y2)-mu(k(y)|x2)>
                    = K_y1y2 - M_y1 - M_y2 + M_y1y2
    M_y1 = K_x1X @ K_XX_inv @ K_y2Y.T
    M_y2 = K_y1Y @ K_XX_inv @ K_x2X.T
    M_y1y2 = K_x1X @ K_XX_inv_K_YY @ K_XX_inv @ K_x2X.T
    Capital X/Y denotes training data used for estimating the conditional mean,
    lowercase x1/y1 and x2/y2 denote input data.

    Args:
        x1 (torch.Tensor): Input data 1 of shape (n1, d_x).
        y1 (torch.Tensor): Output data 1 of shape (n1, d_y).
        x2 (torch.Tensor): Input data 2 of shape (n2, d_x).
        y2 (torch.Tensor): Output data 2 of shape (n2, d_y).
        model_x: Kernel model for x with attributes kernel_matrix and ridge_lambda.
        model_y: Kernel model for y with attribute kernel_matrix.
    Returns:
        K_y1y2_centered (torch.Tensor): Centered cross-kernel matrix of shape (n1, n2). 
    """

    # Compute the kernel matrices for y
    K_y1y2 = model_y(y1, y2).double()
    K_y1Y = model_y(y1).double()
    K_y2Y = model_y(y2).double()
    # Compute the kernel matrices for x
    K_x1X = model_x(x1).double()
    K_x2X = model_x(x2).double()
    # Compute the inverse kernel matrices for X
    K_XX_inv, K_XX_inv_K_YY = solve_regularized_kernel_matrix_system(model_x.kernel_matrix, model_y.kernel_matrix, 
                                                        model_x.ridge_lambda)
    # Compute the matrix M
    M_y1 = K_x1X @ K_XX_inv @ K_y2Y.T
    M_y2 = K_y1Y @ K_XX_inv @ K_x2X.T
    M_y1y2 = K_x1X @ K_XX_inv_K_YY @ K_XX_inv @ K_x2X.T

    # Compute the centered kernel matrix for y
    K_y1y2_centered = K_y1y2 - M_y1 - M_y2 + M_y1y2

    return K_y1y2_centered.detach().float()


def solve_regularized_kernel_matrix_system(K_XX, K_YY, ridge_lambda):
    """
    Solves the regularized kernel system (K_XX + ridge_lambda * I).
    
    Args:
        K_XX (torch.Tensor): Kernel matrix of shape (n, n).
        K_YY (torch.Tensor): Kernel matrix of shape (n, n).
        ridge_lambda (float): Regularization parameter.

    Returns:
        K_XX_inv (torch.Tensor): Inverse of regularized K_XX.
        K_XX_inv_K_YY (torch.Tensor): (K_XX + n * ridge_lambda * I)^(-1) @ K_YY.
    """
    n = K_XX.shape[0]
    K_XX = add_diag(K_XX, n * ridge_lambda).double()
    K_YY = torch.cat((torch.eye(n).to(K_XX.device), K_YY), 1).double()

    W_all = torch.linalg.solve(K_XX, K_YY)

    K_XX_inv = W_all[:, :n]      # (K_XX + n * ridge_lambda*I)^(-1)
    K_XX_inv_K_YY = W_all[:, n:]  # (K_XX + n * ridge_lambda*I)^(-1) K_YY

    return K_XX_inv, K_XX_inv_K_YY


def get_regularized_kernel_matrix_inverse(K_XX, ridge_lambda):
    """
    Computes the inverse of the regularized kernel matrix (K_XX + n * ridge_lambda * I).
    
    Args:
        K_XX (torch.Tensor): Kernel matrix of shape (n, n).
        ridge_lambda (float): Regularization parameter.

    Returns:
        K_XX_inv (torch.Tensor): Inverse of regularized K_XX.
    """
    n = K_XX.shape[0]
    reg_K_XX = add_diag(K_XX, n * ridge_lambda).double()

    # Compute the inverse using linalg.solve
    identity = torch.eye(n, device=reg_K_XX.device, dtype=reg_K_XX.dtype)
    K_XX_inv = torch.linalg.solve(reg_K_XX, identity)

    return K_XX_inv.float()
