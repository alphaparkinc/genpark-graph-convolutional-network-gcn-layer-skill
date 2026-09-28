# genpark-graph-convolutional-network-gcn-layer-skill

Agent Skill implementing **Spectral Graph Convolutional Network (GCN)** with Kipf & Welling self-loop renormalization and localized spatial feature aggregation.

## Architectural Overview
```mermaid
flowchart TD
    A["Raw Adjacency A"] --> Renorm["Add Self-Loops: A_hat = A + I"]
    Renorm --> Degree["Degree Normalization: S = D_hat^-0.5 * A_hat * D_hat^-0.5"]
    X["Node Features X"] & S --> Agg["Spectral Message Aggregation: S * X"]
    Agg & W["Learnable Weights W"] --> Linear["Linear Projection: (S * X) * W"]
    Linear --> Act["Non-linear Activation: ReLU(...)"]
    Act --> Out["Layer Output Embeddings Z"]
```
