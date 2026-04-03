import torch
from torch_geometric.data import Data


class FlexibleGraphBuilder:
    """
    A graph builder that constructs RST-based graphs from processed data items.
    It builds a homogeneous graph with a unified `edge_index` and an `edge_type` tensor,
    suitable for native PyG layers.
    It assumes that a complete metadata object with all possible edge types is provided
    at initialization.
    """

    def __init__(self, config: dict, metadata: dict):
        """
        Initializes the builder with a configuration and a complete dataset-wide metadata.

        Args:
            config (dict): The graph configuration dictionary.
            metadata (dict): A dictionary containing the pre-computed list of all
                             possible edge types and the mapping to integers.
        """
        self.config = config
        self.metadata = metadata
        self.edge_type_mode = self.config.get("edge_type_mode", "tuple")
        # The model is now responsible for creating the correct mapping.
        self.edge_type_to_int = self.metadata["edge_type_to_idx"]

    def build(self, data_item: dict, max_edu_nodes: int | None = None) -> Data:
        """
        Builds a graph with tensor attributes based on the pre-defined metadata.
        """
        graph = Data()

        nodes_info = data_item["nodes"]
        edu_nodes_info = [n for n in nodes_info if n["type"] == "edu_node"]
        if max_edu_nodes is not None and len(edu_nodes_info) > max_edu_nodes:
            edu_nodes_info = edu_nodes_info[:max_edu_nodes]

        valid_edu_ids = {n["id"] for n in edu_nodes_info}
        relation_nodes_info = [n for n in nodes_info if n["type"] == "relation_node"]

        all_nodes_info = edu_nodes_info + relation_nodes_info
        global_to_unified_idx = {node["id"]: i for i, node in enumerate(all_nodes_info)}
        graph.num_nodes = len(all_nodes_info)

        graph._edu_indices = list(range(len(edu_nodes_info)))
        graph._relation_indices = list(range(len(edu_nodes_info), len(all_nodes_info)))
        graph._edu_mapping = {
            str(node["id"]): i for i, node in enumerate(edu_nodes_info)
        }

        all_edge_indices = []
        all_edge_types_numeric = []
        use_nuclearity = self.config.get("rst_use_nuclearity", False)
        use_distinct_reverse = self.config.get("use_distinct_reverse_edges", True)
        node_id_to_info = {node["id"]: node for node in all_nodes_info}

        for parent_global_id, child_global_id in data_item["parent_child_edges"]:
            if (
                parent_global_id not in global_to_unified_idx
                or child_global_id not in global_to_unified_idx
            ):
                continue
            child_node_info = node_id_to_info.get(child_global_id)
            if child_node_info and child_node_info.get("type") == "edu_node":
                if child_global_id not in valid_edu_ids:
                    continue

            parent_local_idx = global_to_unified_idx[parent_global_id]
            child_local_idx = global_to_unified_idx[child_global_id]
            parent_node = node_id_to_info.get(parent_global_id)
            child_node = node_id_to_info.get(child_global_id)
            if not parent_node or not child_node:
                continue

            rel_name_from_data = parent_node["relation"]
            base_rel_name = rel_name_from_data.split("_")[0]

            fwd_rel_str = base_rel_name
            if use_nuclearity:
                nuclearity = parent_node["nuclearity"]
                child_is_left = parent_node.get("left_child_id") == child_global_id
                if nuclearity == "NN":
                    fwd_rel_str = f"{base_rel_name}_N"
                elif nuclearity == "NS":
                    fwd_rel_str = (
                        f"{base_rel_name}_N"
                        if not child_is_left
                        else f"{base_rel_name}_S"
                    )
                elif nuclearity == "SN":
                    fwd_rel_str = (
                        f"{base_rel_name}_S" if child_is_left else f"{base_rel_name}_N"
                    )

            rev_rel_str = f"rev_{fwd_rel_str}" if use_distinct_reverse else fwd_rel_str

            fwd_key = ("relation_node", fwd_rel_str, child_node["type"])
            rev_key = (child_node["type"], rev_rel_str, "relation_node")

            if fwd_key in self.edge_type_to_int:
                all_edge_indices.append([parent_local_idx, child_local_idx])
                all_edge_types_numeric.append(self.edge_type_to_int[fwd_key])

            if rev_key in self.edge_type_to_int:
                all_edge_indices.append([child_local_idx, parent_local_idx])
                all_edge_types_numeric.append(self.edge_type_to_int[rev_key])

        if all_edge_indices:
            graph.edge_index = (
                torch.tensor(all_edge_indices, dtype=torch.long).t().contiguous()
            )
            graph.edge_type = torch.tensor(all_edge_types_numeric, dtype=torch.long)
        else:
            graph.edge_index = torch.empty((2, 0), dtype=torch.long)
            graph.edge_type = torch.empty((0,), dtype=torch.long)

        root_global_id = data_item.get("root_node_ids", [None])[0]
        if root_global_id is not None and root_global_id in global_to_unified_idx:
            graph._root_idx = global_to_unified_idx[root_global_id]
        else:
            graph._root_idx = -1

        # --- BFS for Node Depth Calculation ---
        # Initialize depths with -1 (unreachable)
        node_depths = {node_id: -1 for node_id in global_to_unified_idx.keys()}

        if root_global_id is not None and root_global_id in global_to_unified_idx:
            # Build an adjacency list for BFS traversal (parent -> children)
            adj_list = {}
            for p_id, c_id in data_item["parent_child_edges"]:
                if p_id not in adj_list:
                    adj_list[p_id] = []
                adj_list[p_id].append(c_id)

            queue = [(root_global_id, 0)]
            node_depths[root_global_id] = 0
            visited = {root_global_id}

            while queue:
                curr_id, depth = queue.pop(0)
                if curr_id in adj_list:
                    for child_id in adj_list[curr_id]:
                        if (
                            child_id not in visited
                            and child_id in global_to_unified_idx
                        ):
                            visited.add(child_id)
                            node_depths[child_id] = depth + 1
                            queue.append((child_id, depth + 1))

        # Convert depths to a tensor aligned with node indices
        # Use a list comprehension to ensure order matches global_to_unified_idx values (0, 1, 2...)
        # Since global_to_unified_idx maps ID -> 0..N-1, we can create a list of size N
        sorted_depths = [0] * len(global_to_unified_idx)
        for nid, idx in global_to_unified_idx.items():
            sorted_depths[idx] = node_depths.get(nid, -1)

        graph.node_depth = torch.tensor(sorted_depths, dtype=torch.long)

        return graph
