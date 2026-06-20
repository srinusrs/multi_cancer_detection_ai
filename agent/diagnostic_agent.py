from model.inference import DiagnosticEvaluator
from explainability.shap_explainer import GradientPerturbationExplainer
from .llm_report_generator import ClinicalReportCompiler
from data.synthetic_generator import BiomarkerDataEngine
import numpy as np

class IntegratedDiagnosticAgent:
    """Core Orchestration Agent binding preprocessing, evaluation, and report creation loops."""
    def __init__(self, model, reference_df):
        self.evaluator = DiagnosticEvaluator(model, reference_df)
        self.model = model
        
    def execute_agent_pipeline(self, patient_features: np.ndarray) -> dict:
        # Run Core Evaluation
        eval_metrics = self.evaluator.evaluate_patient_profile(patient_features)
        
        # Compute Attributions
        xai_weights = GradientPerturbationExplainer.compute_attributions(
            self.model, eval_metrics['raw_tensor'], eval_metrics['predicted_class']
        )
        
        # Isolate top 3 feature drivers
        top_indices = np.argsort(np.abs(xai_weights))[::-1][:3]
        top_markers = [BiomarkerDataEngine.MIRNA_PANEL[i] for i in top_indices]
        
        # Map parameters into reporting engines
        report_payload = {
            "status": BiomarkerDataEngine.CLASS_MAP[eval_metrics['predicted_class']],
            "confidence": eval_metrics['confidence'],
            "uncertainty": eval_metrics['uncertainty'],
            "anomaly_score": eval_metrics['anomaly_score'],
            "markers": ", ".join(top_markers)
        }
        
        clinical_report = ClinicalReportCompiler.generate_local_report(report_payload)
        prompt_template = ClinicalReportCompiler.get_prompt_template().format(**report_payload)
        
        return {
            "metrics": eval_metrics,
            "attributions": xai_weights,
            "report": clinical_report,
            "prompt": prompt_template,
            "top_markers": top_markers
        }
