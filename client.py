"""Spectral Graph Convolutional Network (GCN) Layer Engine.
100% Python Standard Library.
"""

import math

class GCNLayer:
    """Spectral Graph Convolutional Network forward propagation with self-loop renormalization."""
    @staticmethod
    def forward(adj_matrix, feature_matrix, weight_matrix):
        n = len(adj_matrix)
        a_hat = [[adj_matrix[i][j] + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        d_hat = [sum(a_hat[i]) for i in range(n)]
        d_inv_sqrt = [1.0 / math.sqrt(d) if d > 0 else 0.0 for d in d_hat]

        s = [[d_inv_sqrt[i] * a_hat[i][j] * d_inv_sqrt[j] for j in range(n)] for i in range(n)]
        d_in = len(feature_matrix[0])
        sx = [[sum(s[i][k] * feature_matrix[k][d] for k in range(n)) for d in range(d_in)] for i in range(n)]

        d_out = len(weight_matrix[0])
        z = [[0.0] * d_out for _ in range(n)]
        for i in range(n):
            for j in range(d_out):
                val = sum(sx[i][k] * weight_matrix[k][j] for k in range(d_in))
                z[i][j] = max(0.0, val)
        return z
