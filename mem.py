import torch, gc

def print_cuda_mem(msg=""):
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"[{msg}] allocated={alloc:.2f} MB   reserved={reserved:.2f} MB")

def tensor_mem_mb(x, name="tensor"):
    if x is None:
        print(f"{name}: None")
        return
    size = x.nelement() * x.element_size() / (1024**2)
    print(f"{name}: {size:.2f} MB   shape={tuple(x.shape)}")

def alive():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            print(obj.shape, obj.dtype, obj.requires_grad)