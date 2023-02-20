def reset_parameters(model):
    """
    1. Reset all parameters of model.
    """
    for layer in model.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()

def fmttime(seconds): return f"{seconds//3600}:{str((seconds//60)%60).zfill(2)}:{str(seconds%60).zfill(2)}"