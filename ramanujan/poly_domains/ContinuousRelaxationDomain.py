import torch
import torch.optim as optim
from .CartesianProductPolyDomain import CartesianProductPolyDomain

class ContinuousRelaxationDomain(CartesianProductPolyDomain):
    """
    Experimental ContinuousRelaxationDomain.
    Relaxes the discrete integer search space for coefficients to the continuous real domain.
    Employs gradient descent optimization (via PyTorch) to minimize the evaluation loss to 
    satisfy necessary algebraic convergence conditions, then applies lattice rounding 
    to snap backward to nearest integers.
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, lr=0.1, epochs=100, *args, **kwargs):
        self.lr = lr
        self.epochs = epochs
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        # Shift the metadata constraints using optimization
        self._run_gradient_descent()
        super()._setup_metadata()

    def _run_gradient_descent(self):
        """
        Differentiable relaxation over the polynomial coefficient constraints.
        This shifts the bounding box evaluated by the generator to the region 
        with the highest gradient optimization density for the Worpitzky convergence condition.
        """
        # We optimize continuous bounds to minimize the distance to the convergence boundary
        a_bounds = torch.tensor(self.a_coef_range, dtype=torch.float32, requires_grad=True)
        b_bounds = torch.tensor(self.b_coef_range, dtype=torch.float32, requires_grad=True)
        
        optimizer = optim.Adam([a_bounds, b_bounds], lr=self.lr)
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            
            # Reparameterization trick: sample from the relaxed continuous bounds defined by means
            a_mean = a_bounds.mean(dim=1)
            b_mean = b_bounds.mean(dim=1)
            
            # Loss: penalty for violating 4*b[0] + a[0]^2 > 0
            # Using index 0 as a proxy for the leading coefficients constraints
            a_lead = a_mean[0] if len(a_mean) > 0 else 1.0
            b_lead = b_mean[0] if len(b_mean) > 0 else 1.0
            
            margin = 4 * b_lead + a_lead**2
            loss = torch.relu(1.0 - margin) # Push margin to be at least 1.0
            
            # Add small regularization to shrink the bounding width slightly 
            loss += 0.05 * (a_bounds[:, 1] - a_bounds[:, 0]).sum()
            loss += 0.05 * (b_bounds[:, 1] - b_bounds[:, 0]).sum()
            
            loss.backward()
            optimizer.step()
            
            # Enforce min <= max during descent avoiding in-place autograd issues
            with torch.no_grad():
                a_bounds.data[:, 0] = torch.min(a_bounds.data[:, 0], a_bounds.data[:, 1] - 0.1)
                b_bounds.data[:, 0] = torch.min(b_bounds.data[:, 0], b_bounds.data[:, 1] - 0.1)
                
        # Snap back to integers
        self.a_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in a_bounds]
        self.b_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in b_bounds]
