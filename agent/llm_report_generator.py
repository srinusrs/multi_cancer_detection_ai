class ClinicalReportCompiler:
    """Transforms raw machine learning outputs into standard non-alarming reports."""
    
    @staticmethod
    def get_prompt_template() -> str:
        return (
            "ROLE DEFINITION: You are a Clinical AI Genomics Expert.\n"
            "MANDATE: Convert the provided numerical risk matrices into an actionable report.\n"
            "SAFETY INSTRUCTION: Do not provide a final diagnosis. Only provide risk interpretations.\n"
            "CRITICAL BREAKDOWN:\n"
            "- Target Risk Target: {status}\n"
            "- Model Confidence Matrix: {confidence:.2f}%\n"
            "- Pipeline Uncertainty Score: {uncertainty:.4f}\n"
            "- High-Risk Marker Intersections: {markers}\n"
        )

    @staticmethod
    def generate_local_report(metrics: dict) -> str:
        """Deterministic fallback engine generating standard reports without API overhead."""
        if metrics['uncertainty'] > 0.16 or metrics['anomaly_score'] > 0.75:
            return (
                "🚨 LAB EXCLUSIONARY NOTIFICATION: INCONCLUSIVE SCREENING RUN\n"
                "----------------------------------------------------------------------\n"
                "STATUS SUMMARY: CANCELLED / RE-SAMPLING MANDATED\n\n"
                f"TECHNICAL EVALUATION: System flagged an elevated runtime uncertainty boundary "
                f"({metrics['uncertainty']:.4f}) or high anomaly index ({metrics['anomaly_score']*100:.1f}%). "
                "The sample configuration does not meet the requirements for statistical classification.\n\n"
                "CLINICAL INTERVENTION PATHWAYS:\n"
                "1. Reject current sequencing file and execute immediate verification procedures.\n"
                "2. Check specimen extraction steps for cellular background noise or contamination."
            )
            
        return (
            "🧬 CLINICAL MULTI-CANCER BIOMARKER REPORT\n"
            "ISSUED BY: AI Diagnostic Agent Architecture\n"
            "----------------------------------------------------------------------\n\n"
            f"SUMMARY EVALUATION: The biomarker pattern analyzer identified expression adjustments "
            f"strongly correlated with [{metrics['status']}]. Model confidence score matches {metrics['confidence']*100:.1f}%.\n\n"
            f"BIOMARKER DRIVER METRICS: High-impact directional fluctuations were isolated across: {metrics['markers']}.\n\n"
            "DIAGNOSTIC PATHWAY RECOMMENDATIONS:\n"
            "• This assessment presents an advanced computational evaluation of liquid biopsy genomic profiles, not a definitive diagnosis.\n"
            "• Correlate findings with targeted secondary staging procedures (e.g., Low-Dose CT imaging scans or digital mammography).\n"
            "• Schedule an entry consultation to evaluate lifestyle risk factors alongside global tracking panels."
        )
