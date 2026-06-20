import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from .transformer import TabularBiomarkerTransformer
from data.synthetic_generator import BiomarkerDataEngine

class SVAProfileDataset(Dataset):
    def __init__(self, df):
        self.features = torch.tensor(df[BiomarkerDataEngine.MIRNA_PANEL].values, dtype=torch.float32)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def execute_training_pipeline(epochs=5, batch_size=32) -> TabularBiomarkerTransformer:
    """Orchestrates compilation and training iterations for the Deep Learning framework."""
    engine = BiomarkerDataEngine()
    base_data = engine.generate_synthetic_cohort(500)
    aug_data = engine.apply_hybrid_augmentation(base_data)
    training_df = pd.concat([base_data, aug_data], ignore_index=True)
    
    model = TabularBiomarkerTransformer()
    loader = DataLoader(SVAProfileDataset(training_df), batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
    return model
