from client import GCNLayer

adj = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
X = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
W = [[0.5, 0.2], [0.1, 0.8]]

Z = GCNLayer.forward(adj, X, W)
print("GCN Layer Forward Output Embeddings:")
for i, row in enumerate(Z):
    print(f"  Node {i}: {[round(v, 4) for v in row]}")
