import torch

def merge_datasets(datasets):
    """
    Merge multiple datasets by concatenating their a, b, c 
    attributes or the tuples returned by __getitem__.
    
    datasets: list of dataset objects, each should have either 
    attributes 'a', 'b', 'c' (and optionally 'a_m', 'b_m') 
    or return tuples (a,b,c) from __getitem__.

    return:
        dict containing 'a','b','c', 'a_m', 'b_m' Tensor
    """
    if all(hasattr(datasets[0], attr) for attr in ['a','b','c']):
        a = torch.cat([d.a for d in datasets], dim=0)
        b = torch.cat([d.b for d in datasets], dim=0)
        c = torch.cat([d.c for d in datasets], dim=0)
        if hasattr(datasets[0], 'a_m') and hasattr(datasets[0], 'b_m'):
            a_m = torch.cat([d.a_m for d in datasets], dim=0)
            b_m = torch.cat([d.b_m for d in datasets], dim=0)
            return {'a': a, 'b': b, 'c': c, 'a_m': a_m, 'b_m': b_m}
        else:
            return {'a': a, 'b': b, 'c': c}
    else:
        # 假设 __getitem__ 返回 tuple (a,b,c)
        a = torch.cat([d[:][0] for d in datasets], dim=0)
        b = torch.cat([d[:][1] for d in datasets], dim=0)
        c = torch.cat([d[:][2] for d in datasets], dim=0)

    return {'a': a, 'b': b, 'c': c}


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        combined = merge_datasets(datasets)
        self.a = combined['a']
        self.b = combined['b']
        self.c = combined['c']
        self.a_m = combined.get('a_m', None) 
        self.b_m = combined.get('b_m', None) 

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.c[idx], self.a_m[idx] if self.a_m is not None else None, self.b_m[idx] if self.b_m is not None else None