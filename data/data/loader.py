import pandas as pd
import numpy as np
from .synthetic_generator import BiomarkerDataEngine

class GenomicLoader:
    """Option A: Preprocesses and aligns external real datasets with structural targets."""
    
    @staticmethod
    def parse_input_matrix(file_buffer) -> pd.DataFrame:
        """Parses external CSV profiles and normalizes them into standard ML panels."""
        try:
            df = pd.read_csv(file_buffer)
            if "label" not in df.columns:
                df["label"] = 0 # Default evaluation label fallback
                
            # Impute or fill any missing microRNA features in the target panel
            for mirna in BiomarkerDataEngine.MIRNA_PANEL:
                if mirna not in df.columns:
                    df[mirna] = 0.0
                    
            aligned_columns = BiomarkerDataEngine.MIRNA_PANEL + ["label"]
            return df[aligned_columns].astype(np.float32)
        except Exception as e:
            raise ValueError(f"Matrix parsing and panel alignment structural failure: {str(e)}")
