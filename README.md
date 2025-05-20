# 📝 Automating Health Equity Research Classification Using Large Language Models with Advanced Prompting Strategies

This repository contains the code for our work *Automating Health Equity Research Classification Using Large Language Models with Advanced Prompting Strategies*.

## 🚀 Getting Started

### Prerequisites

Ensure you have Conda installed or use a virtual environment.


### Installation Steps

```bash
# Clone the repository
git clone https://github.com/Farzad-1996/healthequity
cd healthequity
```

### 1️⃣ For Llama 3.1 8B and GPT-4
To set up the environment and ensure GPU access, run:

```bash
set echo
umask 0027
# Check available GPUs
nvidia-smi

module load gnu10
module load python

# Create and activate virtual environment
python -m venv llama3.1env
source llama3.1env/bin/activate

# Install dependencies
pip install json datetime
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install accelerate transformers numpy unsloth trl
pip install openai==0.27.0

# Verify installations
pip list
```

