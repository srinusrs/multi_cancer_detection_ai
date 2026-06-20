# 🧬 AI Blood microRNA Multi-Cancer Detection Agent

An end-to-end clinical diagnostic simulation pipeline leveraging a continuous Tabular Transformer framework in PyTorch to analyze high-dimensional liquid biopsy microRNA expression profiles.

## 🚀 Quickstart Deployment

### 1. Initialize local directory and activate python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
2. Install core framework requirements:
Bash
pip install -r requirements.txt
3. Launch application dashboard:
Bash
streamlit run app/streamlit_app.py
🧪 Pipeline Layer Verifications
Data Layer: Utilizes continuous Log-Normal distribution metrics to mimic qPCR counts.

Model Layer: Tabular continuous embeddings processed through PyTorch Multi-Head Attention mechanisms.

Explainability: Perturbation-based feature attributions mapping directly to class outputs.

Agent Layer: Evaluates Monte Carlo Dropout variance and Isolation Forest anomaly boundaries to flag inconclusive runs automatically.


---

## 📈 Operational Framework & Verification Walkthrough

The application coordinates data flow, model processing, and explanations across your modules:

1. **Synthetic Generation:** The `BiomarkerDataEngine` sets up a 100-miRNA panel using a Log-Normal distribution ($\mu=1.5, \sigma=0.3$) to simulate realistic continuous gene expression data. It injects specific upregulation or downregulation patterns depending on the cancer type.
2. **Tabular Transformer:** Instead of using standard ML vectors, `TabularBiomarkerTransformer` isolates every continuous variable, passing it through its own linear projection layer to build a sequence of feature embeddings ($D=32$). It then prepends a learnable classification token (`cls_token`) before routing the tensors through standard Multi-Head Attention transformer blocks.
3. **Uncertainty & Anomaly Scoring:** During inference, the system calculates two safety metrics alongside base probabilities. It monitors prediction stability using Monte Carlo Dropout variance across multiple forward passes, and evaluates structural distance relative to baseline populations via an Isolation Forest anomaly model.
4. **Clinical Agent Report Loop:** The `IntegratedDiagnosticAgent` compiles these para
