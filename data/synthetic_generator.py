import numpy as np
import pandas as pd

class BiomarkerDataEngine:
    """Option B & C: Biological expression profiling synthesizer with hybrid augmentation mechanics."""
    
    # Core 100-miRNA Panel Profile mapping prominent oncological indicators
    MIRNA_PANEL = [
        "hsa-miR-21-5p", "hsa-miR-155-5p", "hsa-miR-210-3p", "hsa-miR-34a-5p", 
        "hsa-miR-10b-5p", "hsa-miR-92a-3p", "hsa-miR-141-3p", "hsa-miR-200c-3p",
        "hsa-miR-29a-3p", "hsa-miR-221-3p", "hsa-miR-222-3p", "hsa-miR-126-3p",
        "hsa-miR-145-5p", "hsa-miR-143-3p", "hsa-miR-19a-3p", "hsa-miR-18a-5p",
        "hsa-miR-125b-5p", "hsa-miR-146a-5p", "hsa-miR-15a-5p", "hsa-miR-16-5p"
    ] + [f"hsa-miR-gen-{i}" for i in range(80)]

    CLASS_MAP = {0: "Normal", 1: "Lung Cancer", 2: "Breast Cancer", 3: "Colon Cancer"}

    @classmethod
    def generate_synthetic_cohort(cls, num_samples=400, seed=42) -> pd.DataFrame:
        """Generates structured cohorts using log-normal baselines to mirror qPCR/Seq configurations."""
        np.random.seed(seed)
        data = []
        samples_per_class = num_samples // 4
        
        for class_idx in range(4):
            for _ in range(samples_per_class):
                # Standard log-normal biological distribution baseline
                profile = np.random.lognormal(mean=1.5, sigma=0.3, size=len(cls.MIRNA_PANEL))
                
                # Apply target biomarker signatures per oncological path
                if class_idx == 1:    # Lung Cancer Vector
                    profile[0] *= 4.2  # Up-regulate miR-21
                    profile[1] *= 3.5  # Up-regulate miR-155
                    profile[2] *= 5.0  # Up-regulate miR-210 (Hypoxia indicator)
                elif class_idx == 2:  # Breast Cancer Vector
                    profile[4] *= 5.5  # Up-regulate miR-10b
                    profile[0] *= 2.8  # Up-regulate miR-21
                    profile[3] *= 0.15 # Down-regulate miR-34a (Tumor Suppressor)
                elif class_idx == 3:  # Colon Cancer Vector
                    profile[5] *= 6.0  # Up-regulate miR-92a
                    profile[6] *= 4.5  # Up-regulate miR-141
                    profile[12] *= 0.1 # Down-regulate miR-145
                    
                data.append(list(profile) + [class_idx])
                
        df = pd.DataFrame(data, columns=cls.MIRNA_PANEL + ["label"])
        return df.astype(np.float32)

    @classmethod
    def apply_hybrid_augmentation(cls, df: pd.DataFrame, noise_level=0.05, mask_prob=0.05) -> pd.DataFrame:
        """Executes Option C hybrid augmentations via Gaussian injection and feature masking."""
        augmented_features = df[cls.MIRNA_PANEL].values.copy()
        
        # 1. Inject Gaussian Noise variance
        noise = np.random.normal(0, noise_level, augmented_features.shape)
        augmented_features += noise
        
        # 2. Apply random feature masking (simulating sequencing dropouts)
        mask = np.random.binomial(1, mask_prob, augmented_features.shape)
        augmented_features[mask == 1] = 0.0
        
        # 3. Apply operational scaling variations
        scaling_factors = np.random.uniform(0.95, 1.05, size=(augmented_features.shape[0], 1))
        augmented_features *= scaling_factors
        
        augmented_df = pd.DataFrame(augmented_features, columns=cls.MIRNA_PANEL)
        augmented_df["label"] = df["label"].values
        return augmented_df.astype(np.float32)
