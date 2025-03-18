# Example test script
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from configuration.xai_config import FederatedXAIConfig
from server.orchestrator import FederatedOrchestrator
import numpy as np
import pandas as pd

# Create simple synthetic data
feature_names = [f'feature_{i}' for i in range(10)]
X = pd.DataFrame(np.random.random((100, 10)), columns=feature_names)
y = pd.Series((X.sum(axis=1) > 5).astype(int))

# Create very simple XAI config
xai_config = FederatedXAIConfig(
    collect_explanations=True,
    explanation_rounds=[1, 2]  # Only collect on rounds 1 and 2
)

# Create orchestrator with XAI config
orchestrator = FederatedOrchestrator(
    input_dim=10,
    features=feature_names,
    xai_config=xai_config
)

# Add a few clients
for i in range(3):
    idx = np.random.choice(len(X), 50, replace=False)
    orchestrator.add_client(f'client_{i+1}', X.iloc[idx], y.iloc[idx])

# Train for just 2 rounds to test
history = orchestrator.train_federated(
    num_rounds=2,
    local_epochs=1,
    min_clients=2
)

# Check if explanations were collected
print(f"Explanations collected: {len(orchestrator.explanation_history)}")