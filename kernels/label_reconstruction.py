
import torch

def llg_label_recon(grads, b):

    '''
    b: batch size
    '''
    # 计算m
    S = 0
    g = grads[-2].sum(-1)

    idx = torch.where(g < 0)[0] # 获得小于0的索引
    m = (g[idx].sum() / b) *(1 + 1 / len(idx)) 

    # 获得标签
    lbls = []
    for i, gi in enumerate(g):
        if gi < 0:
            lbls.append(i)
            g[i] -= m
    g -= S
    while len(lbls) < b:
        j = g.argmin().item()
        lbls.append(j)
        g[j] -= m

    return torch.as_tensor(lbls)