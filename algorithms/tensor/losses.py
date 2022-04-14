import torch


def L1Loss_masked(x, y, mask, reduction='mean'):
    flat_mask = torch.nn.Flatten()(mask)
    count = torch.count_nonzero(flat_mask, 1)

    if reduction == 'mean':
        return torch.sum(torch.abs((x - y) * mask)) / torch.sum(count)
    elif reduction == 'sum':
        return torch.sum(torch.abs((x - y) * mask))
    else:
        return torch.sum(torch.abs((x - y) * mask)) / torch.sum(count)
