import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import RGCNConv, GCNConv, global_mean_pool
from transformers.models.auto.modeling_auto import AutoModel

from utils.flexible_graph_builder import FlexibleGraphBuilder
from .modules import ClassificationHead


class RACEModel(nn.Module):
    """
    RACE (Rhetorical Analysis for Creator-Editor Modeling):
    An RST-based Graph Neural Network model for fine-grained LLM-generated text detection.

    This model converts a document's RST (Rhetorical Structure Theory) tree into a heterogeneous
    graph, then applies relational GNN layers (RGCN) to learn structure-aware representations,
    and finally classifies the document based on the root node's representation.

    Architecture:
        1. Backbone (RoBERTa) encodes the document and produces contextual EDU features.
        2. RST-based graph is constructed with EDU nodes and Relation nodes.
        3. RGCN layers perform relational message passing over the graph.
        4. Root pooling extracts the document-level representation.
        5. MLP classifier produces the final prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        gnn_hidden_dim: int,
        num_heads: int,
        num_classes: int,
        config: dict,
        metadata: dict,
        output_features: bool = False,
    ):
        super().__init__()
        # --- 1. Module Setup and Configuration ---
        self.config = config
        self.metadata = metadata
        self.output_features = output_features

        # GNN and feature dimensions.
        self.bert_dim = config.get("bert_dim", 768)
        self.feature_dim = feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_layers = config.get("num_gnn_layers", 2)

        # --- 2. Backbone Model (RoBERTa) ---
        self.bert_model = AutoModel.from_pretrained(config["backbone_model_path"])
        print(f"Loading backbone model from {config['backbone_model_path']}")

        # Freezing logic for the backbone model's layers based on the config.
        unfreeze_layers = config.get("unfreeze_bert_layers", 0)
        if unfreeze_layers > 0 and hasattr(self.bert_model, "encoder"):
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfreeze_layers == -1:  # Unfreeze all layers
            self.bert_model.requires_grad_(True)
        else:  # Freeze all layers
            self.bert_model.requires_grad_(False)

        # --- 3. Relation Mapping (Critical for RGCN) ---
        # Maps edge type tuples like ('edu_node', 'Elaboration', 'relation_node') to unique integer IDs.
        # This is required for the `edge_type` tensor used by native relational GNN layers.
        self.edge_types_list = list(metadata["edge_types"])
        self.edge_type_to_idx = {
            etype: i for i, etype in enumerate(self.edge_types_list)
        }
        self.num_relations = len(self.edge_types_list)
        self.metadata["edge_type_to_idx"] = self.edge_type_to_idx
        print(
            f"Initialized RACE model with {self.num_relations} relation types."
        )
        self.graph_builder = FlexibleGraphBuilder(config, metadata)

        # --- 4. Node Type Embeddings ---
        # Creates an embedding layer to distinguish original node types (0 for EDU, 1 for Relation).
        # This allows the model to learn type-specific information even after node features are merged.
        self.node_type_embedding = nn.Embedding(2, self.bert_dim)

        # --- 5. Unified Projection Layer ---
        # Projects the initial node features (BERT output + type embedding) into the GNN's working dimension.
        # The LayerNorm is crucial for stabilizing training by normalizing feature distributions.
        self.node_proj = nn.Sequential(
            nn.Linear(self.bert_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(config.get("proj_dropout", 0.1)),
        )

        # --- 6. Native Relational GNN Layers ---
        # Uses efficient, native PyG layers designed for relational graphs.
        self.convs = nn.ModuleList()
        gnn_layer_type = self.config.get("gnn_layer_type", "RGCN")
        for i in range(self.num_layers):
            in_dim = self.feature_dim if i == 0 else self.gnn_hidden_dim
            if gnn_layer_type == "GCN":
                self.convs.append(
                    GCNConv(
                        in_channels=in_dim,
                        out_channels=gnn_hidden_dim,
                    )
                )
            else:  # Default to RGCN
                self.convs.append(
                    RGCNConv(
                        in_channels=in_dim,
                        out_channels=gnn_hidden_dim,
                        num_relations=self.num_relations,
                        num_bases=config.get("rgcn_num_bases", None),
                    )
                )

        # --- 7. Classifier ---
        classifier_type = self.config.get("classifier_type", "linear")
        if classifier_type == "mlp":
            self.classifier = ClassificationHead(
                input_dim=self.gnn_hidden_dim,
                hidden_dim=self.gnn_hidden_dim,
                num_classes=num_classes,
                dropout_prob=self.config.get("classifier_dropout", 0.1),
            )
        else:
            self.classifier = nn.Linear(self.gnn_hidden_dim, num_classes)

        # --- 8. Weight Initialization ---
        self.node_proj.apply(self._init_weights)
        self.convs.apply(self._init_weights)
        self._init_classifier(self.classifier)

    def _init_weights(self, m):
        """Initializes weights for Linear and LayerNorm layers using Xavier uniform."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_classifier(self, module):
        """Initializes classifier weights with small normal distribution for stable initial predictions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "children"):
            for child in module.children():
                self._init_classifier(child)

    def forward(self, batch: dict, batch_idx: int = -1):
        """
        Forward pass of the RACE model.

        Args:
            batch (dict): A batch dictionary from the DataLoader containing:
                - 'full_text_input_ids': Token IDs for the full document text.
                - 'full_text_attention_mask': Attention mask for the full document text.
                - 'edu_token_spans': List of (start, end) token spans for each EDU.
                - 'nodes': List of node info dicts for each item.
                - 'parent_child_edges': List of (parent_id, child_id) edges for each item.
                - 'root_node_ids': List of root node IDs for each item.
                - 'descendant_map': Map from relation node ID to its descendant EDU IDs.
            batch_idx (int): Index of the current batch (used for debugging).

        Returns:
            If output_features is True: dict with 'features' and 'logits'.
            Otherwise: logits tensor of shape [batch_size, num_classes].
        """
        device = self.node_proj[0].weight.device

        # --- Part 1: Initial Feature and Graph Preparation ---
        # Get contextualized token embeddings from the backbone model.
        input_ids = batch["full_text_input_ids"].to(device)
        attention_mask = batch["full_text_attention_mask"].to(device)
        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = bert_outputs.last_hidden_state
        batch_size = last_hidden_state.size(0)

        # Build a pre-homogenized `Data` object for each item in the batch.
        graph_list = [
            self.graph_builder.build(
                {
                    "nodes": batch["nodes"][i],
                    "parent_child_edges": batch["parent_child_edges"][i],
                    "root_node_ids": batch["root_node_ids"][i],
                },
                max_edu_nodes=len(batch["edu_token_spans"][i]),
            )
            for i in range(batch_size)
        ]

        graph_batch = Batch.from_data_list(graph_list).to(device)

        # Calculate initial EDU features by averaging token embeddings for each span.
        edu_features_per_item = [
            (
                torch.stack(
                    [
                        (
                            last_hidden_state[i][start:end].mean(dim=0)
                            if start < end
                            else torch.zeros(self.bert_dim, device=device)
                        )
                        for start, end in batch["edu_token_spans"][i]
                    ]
                )
                if batch["edu_token_spans"][i]
                else torch.empty((0, self.bert_dim), device=device)
            )
            for i in range(batch_size)
        ]

        # ==============================================================================
        # Part 2: GNN Input Preparation
        # ==============================================================================

        # 1. Create a placeholder for the unified feature tensor.
        unified_features = torch.zeros(
            (graph_batch.num_nodes, self.bert_dim), device=device
        )
        root_node_indices = []

        # 2. Loop through each item to calculate its features and place them in the unified tensor.
        for i in range(batch_size):
            graph_item = graph_list[i]
            edu_feats = edu_features_per_item[i]

            # Calculate initial features for relation nodes (average of descendant EDUs).
            num_rels = len(graph_item._relation_indices)
            relation_features = torch.zeros((num_rels, self.bert_dim), device=device)
            if num_rels > 0:
                desc_map = batch["descendant_map"][i]
                rel_nodes = [
                    n for n in batch["nodes"][i] if n["type"] == "relation_node"
                ]
                for rel_idx, rel_node_info in enumerate(rel_nodes):
                    desc_edu_indices = [
                        graph_item._edu_mapping[str(eid)]
                        for eid in desc_map.get(rel_node_info["id"], [])
                        if str(eid) in graph_item._edu_mapping
                    ]
                    if desc_edu_indices and edu_feats.numel() > 0:
                        relation_features[rel_idx] = edu_feats[desc_edu_indices].mean(
                            dim=0
                        )

            # 3. Use `ptr` (batch offsets) to find the correct slice in the batch tensor and fill features.
            offset = graph_batch.ptr[i].item()
            # Place EDU features and add type embedding 0.
            edu_indices = torch.tensor(graph_item._edu_indices, device=device) + offset
            unified_features[edu_indices] = edu_feats + self.node_type_embedding(
                torch.zeros(len(edu_indices), dtype=torch.long, device=device)
            )
            # Place relation features and add type embedding 1.
            if num_rels > 0:
                rel_indices = (
                    torch.tensor(graph_item._relation_indices, device=device) + offset
                )
                unified_features[rel_indices] = (
                    relation_features
                    + self.node_type_embedding(
                        torch.ones(len(rel_indices), dtype=torch.long, device=device)
                    )
                )

            # Record the absolute index of the root node in the batch.
            if graph_item._root_idx != -1:
                root_node_indices.append(graph_item._root_idx + offset)

        # 4. Project features and get graph structure directly from the `Data` batch object.
        x = self.node_proj(unified_features)
        edge_index, edge_type, batch_vector = (
            graph_batch.edge_index,
            graph_batch.edge_type,
            graph_batch.batch,
        )

        # ==============================================================================
        # Part 3: GNN Propagation, Pooling & Classification
        # ==============================================================================
        # 1. Run the GNN layers.
        gnn_layer_type = self.config.get("gnn_layer_type", "RGCN")
        for i, conv in enumerate(self.convs):
            if gnn_layer_type == "GCN":
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.elu(x)  # Apply activation between layers.

        # 2. Pool node features to get a single vector for each graph.
        pooling_strategy = self.config.get("pooling_strategy", "root")
        if pooling_strategy == "root" and root_node_indices:
            # Use precise root indices if available.
            pooled_features = x[root_node_indices]
        else:
            # Fallback to global mean pooling over all nodes in each graph.
            pooled_features = global_mean_pool(x, batch_vector)

        # Safety check for empty graphs in a batch.
        if pooled_features.size(0) != batch_size:
            raise ValueError(
                f"Pooled features size {pooled_features.size(0)} does not match batch size {batch_size}."
            )

        # 3. Final classification.
        logits = self.classifier(pooled_features)

        if self.output_features:
            return {"features": pooled_features, "logits": logits}
        else:
            return logits


class FlexibleBaselineModel(nn.Module):
    """
    Baseline model for ablation study: only includes the backbone model and a classifier.
    No GNN components or graph building. Uses the [CLS] token representation directly.
    """

    def __init__(
        self,
        feature_dim: int,
        gnn_hidden_dim: int,
        num_heads: int,
        num_classes: int,
        config: dict,
        metadata: dict = None,
        output_features: bool = False,
    ):
        super().__init__()
        self.config = config
        self.output_features = output_features
        self.bert_dim = config.get("bert_dim", 768)

        # 1. Backbone Model
        self.bert_model = AutoModel.from_pretrained(config["backbone_model_path"])
        print(f"Loading baseline backbone model from {config['backbone_model_path']}")

        # Freezing logic (identical to RACEModel)
        unfreeze_layers = config.get("unfreeze_bert_layers", 0)
        if unfreeze_layers > 0 and hasattr(self.bert_model, "encoder"):
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif unfreeze_layers == -1:
            self.bert_model.requires_grad_(True)
        else:
            self.bert_model.requires_grad_(False)

        # 2. Classifier
        classifier_type = self.config.get("classifier_type", "linear")
        if classifier_type == "mlp":
            self.classifier = ClassificationHead(
                input_dim=self.bert_dim,
                hidden_dim=self.bert_dim,
                num_classes=num_classes,
                dropout_prob=self.config.get("classifier_dropout", 0.1),
            )
        else:
            self.classifier = nn.Linear(self.bert_dim, num_classes)

        # 3. Weight Initialization
        self._init_classifier(self.classifier)

    def _init_classifier(self, module):
        """Initializes classifier weights with small normal distribution for stable initial predictions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "children"):
            for child in module.children():
                self._init_classifier(child)

    def forward(self, batch: dict, batch_idx: int = -1):
        """
        Forward pass of the baseline model.
        Uses only the [CLS] token from the backbone for classification, without any GNN.

        Args:
            batch (dict): A batch dictionary from the DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            If output_features is True: dict with 'features' and 'logits'.
            Otherwise: logits tensor of shape [batch_size, num_classes].
        """
        # Determine device from the classifier layer.
        device = next(self.classifier.parameters()).device

        input_ids = batch["full_text_input_ids"].to(device)
        attention_mask = batch["full_text_attention_mask"].to(device)

        bert_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Standard BERT pooling: take the [CLS] token (index 0)
        pooled_output = bert_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        if self.output_features:
            return {"features": pooled_output, "logits": logits}
        else:
            return logits
