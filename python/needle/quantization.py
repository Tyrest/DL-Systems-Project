import numpy as np

def compute_scale_zero_point(min_val, max_val, num_bits=8, symmetric=False):
    """
    Compute scale and zero_point for quantization.
    """
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    
    if symmetric:
        # For symmetric quantization, we force zero_point to be 0 (or mid-point)
        # But typically for int8 symmetric, we map [-max_abs, max_abs] to [-127, 127]
        # Here we are using uint8 range [0, 255] or int8 [-128, 127].
        # The plan mentions int8, so let's assume signed int8 [-128, 127].
        # However, standard quantization often uses uint8 for activations and int8 for weights.
        # The plan says "int8 data types". Let's stick to signed int8 [-128, 127].
        qmin = -128.
        qmax = 127.
        max_abs = max(abs(min_val), abs(max_val))
        min_val = -max_abs
        max_val = max_abs
    else:
        # Asymmetric quantization
        # We want to map [min_val, max_val] to [qmin, qmax]
        # But we are using int8, so range is [-128, 127]
        qmin = -128.
        qmax = 127.

    # Extend the range to include 0
    min_val = min(min_val, 0.)
    max_val = max(max_val, 0.)

    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        scale = 1.0

    zero_point = qmin - min_val / scale
    
    # Clamp zero_point to valid range and round
    zero_point = max(qmin, min(qmax, round(zero_point)))
    
    return float(scale), int(zero_point)


class Observer:
    def update(self, x):
        raise NotImplementedError()
    
    def get_qparams(self):
        raise NotImplementedError()

class MinMaxObserver(Observer):
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update(self, x):
        # x is a numpy array or Tensor
        if hasattr(x, 'numpy'):
            x = x.numpy()
        self.min_val = min(self.min_val, x.min())
        self.max_val = max(self.max_val, x.max())
        
    def get_qparams(self):
        return compute_scale_zero_point(self.min_val, self.max_val)

def calibrate(model, dataloader, device=None):
    """
    Run the model on the dataloader to collect statistics for quantization.
    This assumes the model has been instrumented with observers.
    """
    model.eval()
    # Enable calibration mode
    # We need to traverse the model and set a flag or ensure observers are active
    # For this simple implementation, we assume the model's forward pass 
    # will update observers if they exist.
    
    # We need to make sure we are in a mode where observers are updated.
    # Let's assume we add a 'calibration_mode' to the model or layers.
    # Or we can just rely on the fact that if observers are present, we update them.
    
    # But wait, we only want to update observers during calibration, not inference.
    # So we should probably have a context manager or a flag.
    # For now, let's just iterate.
    
    # We need to tell the model to update observers.
    # Let's assume we add a method `model.reset_observers()` and `model.calibrate()`?
    # The plan says: "Update forward(): Calibration Mode: If enabled, update the observer..."
    
    # So we will set a flag on the model.
    if hasattr(model, 'calibration_mode'):
        model.calibration_mode = True
        
    # Also need to set it for all submodules
    for module in model.modules():
        if hasattr(module, 'calibration_mode'):
            module.calibration_mode = True
            
    for batch in dataloader:
        x, _ = batch
        if device:
            x = x.to(device)
        model(x)
        
    # Disable calibration mode
    if hasattr(model, 'calibration_mode'):
        model.calibration_mode = False
    for module in model.modules():
        if hasattr(module, 'calibration_mode'):
            module.calibration_mode = False
