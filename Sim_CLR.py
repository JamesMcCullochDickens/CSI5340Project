import torch
import numpy as np
import torch.nn.functional as F

# assuming the batch elements are ordered as
# 2N elements, adjacent elements are from the same image
# assuming x is of shape (2N, D)$
def sim_clr_loss(x, device, with_normalization=True, temperature=1.0):
    N = x.shape[0]
    if with_normalization:
        x = F.normalize(x, p=2, dim=1) # normalize the data
    cos_sim = torch.matmul(x, x.permute(1, 0))/temperature
    exp_sim = torch.exp(cos_sim)

    loss_matrix1 = torch.ones(N, N)
    loss_matrix1.fill_diagonal_(0.0)

    block_val = torch.tensor([[0, 1], [1, 0]])
    blocks = []
    for i in range(int(N/2)):
        blocks.append(block_val)
    loss_matrix2 = torch.block_diag(*blocks).float()

    denominator = torch.matmul(exp_sim, loss_matrix1)

    numerator = torch.matmul(exp_sim, loss_matrix2)
    loss_matrix = torch.div(numerator, denominator)
    return torch.mean(loss_matrix)

"""
# testing sim_clr_loss for N=2, D=2
N = 2
D = 2
x = torch.rand(2*N, D)
loss = sim_clr_loss(x, True, 1.0)
debug = "debug"
"""


def row_wise_product(A, B):
    num_rows, num_cols = A.shape[0], A.shape[1]
    prod = torch.bmm(A.view(num_rows, 1, num_cols), B.view(num_rows, num_cols, 1))
    return prod

"""
# testing row_wise_product, should give [3, 7]
A = torch.tensor(np.array([[1, 2], [3, 4]]))
B = torch.tensor(np.array([[1, 1], [1, 1]]))
C = row_wise_product(A, B)
debug = "debug"
"""




