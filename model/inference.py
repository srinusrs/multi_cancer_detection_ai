import torch
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest
from data.synthetic_generator import BiomarkerDataEngine

class DiagnosticEvaluator:
    """Manages multi-class prediction distributions, uncertainty calculations, and anomalies."""
    def __init__(self, model, baseline_dataframe):
        self.model = model
        self.anomaly_engine = IsolationForest(contamination=0.05, random_state=42)
        self.anomaly_engine.fit(baseline_dataframe[BiomarkerDataEngine.MIRNA_PANEL].values)
        
    def evaluate_patient_profile(self, feature_array: np.ndarray):
        self.model.eval()
        tensor_in = torch.tensor(feature_array, dtype=torch.float32).view(1, -1)
        
        # 1. Base Class Probability distributions
        with torch.no_grad():
            raw_logits = self.model(tensor_in)
            probs = F.softmax(raw_logits, dim=-1).squeeze().numpy()
            
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        # 2. Monte Carlo Dropout Loop for true Uncertainty tracking
        mc_iterations = 20
        mc_results = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.TransformerEncoderLayer):
                module.train() # Inject model execution variance
                
        with torch.no_grad():
            for _ in range(mc_iterations):
                mc_results.append(F.softmax(self.model(tensor_in), dim=-1).squeeze().numpy())
                
        uncertainty = float(np.mean(np.std(np.array(mc_results), axis=0)))
        
        # 3. Isolation Anomaly Verification
        if_score = self.anomaly_engine.decision_function(feature_array.reshape(1, -1))[0]
        anomaly_score = float(1.0 - (if_score + 1.0) / 2.0)
        
        return {
            "probabilities": probs,
            "predicted_class": pred_class,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "anomaly_score": anomaly_score,
            "raw_tensor": tensor_in
        }
