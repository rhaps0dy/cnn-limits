from experiments.save_new import *

## TORCH model
@experiment.capture
def torch_model(dataset_name):
    import sys
    import importlib
    sys.path.append("/home/ag919/Programacio/cnn-gp")
    config = importlib.import_module(f"configs.{dataset_name.lower()}")
    return config.initial_model.cuda()

def torch_kernel_fn(kernel_fn):
    import torch
    def kern(x1, x2, same, diag):
        with torch.no_grad():
            return (kernel_fn(x1.cuda(), x2.cuda(), same, diag)
                    .detach().cpu().numpy())
    return kern

@experiment.command
def test_kernels():
    train_set, _ = load_dataset()
    loader = iter(torch.utils.data.DataLoader(train_set, batch_size=4))
    x1, _ = next(loader)
    x2, _ = next(loader)

    torch_kern = torch_kernel_fn(torch_model())
    print("Torch:", torch_kern(x1, x2, False, False))

    _, _, kernel_fn = jax_model()
    kern = jitted_kernel_fn(kernel_fn)
    print("Jax:", kern(x1, x2, False, False))

@experiment.command
def benchmark_main(worker_rank):
    train_set, test_set = load_dataset()
    train_set = Subset(train_set, range(1000))
    torch_kern = torch_kernel_fn(torch_model())
    save_K(torch_kern,     kern_name="Kxx_torch",     X=train_set, X2=None,      diag=False)

    _, _, kernel_fn = jax_model()
    kern = jitted_kernel_fn(kernel_fn)
    save_K(kern,     kern_name="Kxx",     X=train_set, X2=None,      diag=False)

if __name__ == '__main__':
    experiment.run_commandline()
