import torch
import torch.nn.functional as F
import numpy as np
from data.synthetic_generator import BiomarkerDataEngine

class GradientPerturbationExplainer:
    """Direct implementation of Perturbation Attribution to map feature vectors securely."""
    @staticmethod
    def compute_attributions(model, sample_tensor, target_class, steps=15) -> np.ndarray:
        model.eval()
        num_features = len(BiomarkerDataEngine.MIRNA_PANEL)
        attributions = np.zeros(num_features)
        working_batch = sample_tensor.clone().repeat(steps, 1)
        
        for idx in range(num_features):
            base_value = sample_tensor[0, idx].item()
            if base_value == 0: continue
            
            # Construct a swept parameter space across standard ranges
            sweep_values = np.linspace(base_value * 0.2, base_value * 2.0, steps)
            for s in range(steps):
                working_batch[s, idx] = sweep_values[s]
                
            with torch.no_grad():
                outputs = F.softmax(model(working_batch), dim=-1)
                probs = outputs[:, target_class].cpu().numpy()
                
            # Extract first-order derivative trend line vectors
            gradient_slope = np.polyfit(sweep_values, probs, 1)[0]
            attributions[idx] = gradient_slope * base_value
            
        return attributions
