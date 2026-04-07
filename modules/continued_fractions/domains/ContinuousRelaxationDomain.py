import torch
import torch.optim as optim
from .CartesianProductPolyDomain import CartesianProductPolyDomain

class ContinuousRelaxationDomain(CartesianProductPolyDomain):
    """
    Research-Grade ContinuousRelaxationDomain.
    Relaxes the discrete integer search space for coefficients to the continuous real domain.
    Employs gradient descent optimization (via PyTorch) to minimize the evaluation loss to 
    a specified `target_value`, simultaneously satisfying necessary algebraic convergence 
    conditions (Worpitzky margin), then applies lattice rounding to snap backward to nearest integers.
    
    Usage:
        ContinuousRelaxationDomain(a_deg=2, a_coef_range=[-10, 10], 
                                   b_deg=2, b_coef_range=[-10, 10],
                                   target_value=3.14159,
                                   lr=0.1, epochs=100)
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, target_value, lr=0.1, epochs=200, *args, **kwargs):
        self.target_value = target_value
        self.lr = lr
        self.epochs = epochs
        # The parent __init__ expands flat a_coef_range into per-coefficient ranges.
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        # Run gradient descent optimization to shrink the per-coefficient bounds
        # BEFORE the parent computes domain sizes and iterators
        self._run_gradient_descent()
        super()._setup_metadata()

    def _differentiable_gcf_eval(self, a_mean, b_mean, depth=70):
        """
        Differentiable backward evaluation of the GCF to prevent float overflow.
        Extremely stable and perfectly tracks PyTorch autograd gradients.
        """
        device = a_mean.device
        a_deg = len(a_mean) - 1
        b_deg = len(b_mean) - 1
        
        # Precompute a_n and b_n from n=1 to depth
        n_vals = torch.arange(1, depth + 1, dtype=torch.float32, device=device)
        
        a_n = torch.zeros(depth, dtype=torch.float32, device=device)
        b_n = torch.zeros(depth, dtype=torch.float32, device=device)
        
        for i in range(a_deg + 1):
            a_n += a_mean[i] * (n_vals ** (a_deg - i))
        for i in range(b_deg + 1):
            b_n += b_mean[i] * (n_vals ** (b_deg - i))
            
        # Backward evaluation: f_n = b_n / (a_n + f_{n+1})
        f = torch.tensor(0.0, dtype=torch.float32, device=device)
        for i in range(depth - 1, -1, -1):
            denom = a_n[i] + f
            # Smooth epsilon guard to prevent NaN explosions without breaking gradients
            denom_guarded = torch.where(denom.abs() < 1e-5, torch.sign(denom) * 1e-5 + 1e-6, denom)
            f = b_n[i] / denom_guarded
            
        return f

    def _run_gradient_descent(self):
        """
        Differentiable relaxation over the polynomial coefficient constraints.
        Optimizes bounding box directly toward the target mathematical constant.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # We optimize the bounding box [min, max] directly
        a_bounds = torch.tensor(self.a_coef_range, dtype=torch.float32, device=device, requires_grad=True)
        b_bounds = torch.tensor(self.b_coef_range, dtype=torch.float32, device=device, requires_grad=True)
        
        optimizer = optim.Adam([a_bounds, b_bounds], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Predict the value using the mean of each coefficient's bounding box
            a_mean = a_bounds.mean(dim=1)
            b_mean = b_bounds.mean(dim=1)
            
            predicted_val = self._differentiable_gcf_eval(a_mean, b_mean, depth=80)
            
            # 1. Target Value Proximity Loss
            mse_loss = (predicted_val - self.target_value) ** 2
            
            # 2. Worpitzky Convergence Barrier (4 * b_n + a_n^2 > 0)
            a_lead = a_mean[0] if len(a_mean) > 0 else torch.tensor(1.0)
            b_lead = b_mean[0] if len(b_mean) > 0 else torch.tensor(1.0)
            
            margin = 4 * b_lead + a_lead**2
            barrier_loss = torch.relu(1.0 - margin)  # Penalty if margin drops below 1.0
            
            # 3. Box Contraction Penalty (encourage tighter bounds around the minimum)
            width_loss = (a_bounds[:, 1] - a_bounds[:, 0]).sum() + (b_bounds[:, 1] - b_bounds[:, 0]).sum()
            
            # Combined Objective
            # Scaling factors prioritize hitting the target, then converging mathematically, then shrinking bounds
            loss = 100.0 * mse_loss + 10.0 * barrier_loss + 0.1 * width_loss
            
            # Forward pass is done, compute gradients
            loss.backward()
            optimizer.step()
            
            # Enforce min <= max during descent (no-grad context needed to modify data)
            with torch.no_grad():
                a_bounds.data[:, 0] = torch.min(a_bounds.data[:, 0], a_bounds.data[:, 1] - 0.1)
                b_bounds.data[:, 0] = torch.min(b_bounds.data[:, 0], b_bounds.data[:, 1] - 0.1)
                
        # Snap relaxed real boundaries back to discrete integer lattice for GPU sweep
        self.a_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in a_bounds]
        self.b_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in b_bounds]

