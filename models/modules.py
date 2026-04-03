import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    A more advanced classification head, similar to the one used in RoBERTa.
    It consists of a two-layer MLP with a non-linear activation and dropout.

    Architecture: Linear -> Activation -> Dropout -> Linear(out)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_prob: float = 0.1):
        """
        Args:
            input_dim (int): Dimension of the input features (e.g., gnn_hidden_dim).
            hidden_dim (int): Dimension of the intermediate hidden layer.
            num_classes (int): Number of output classes.
            dropout_prob (float): Dropout probability.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for the classification head.
        """
        x = self.dropout(x) # Dropout on the pooled features first
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
