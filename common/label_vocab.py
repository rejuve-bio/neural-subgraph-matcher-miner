import json
import os

import networkx as nx

VOCAB_SCHEMA_VERSION = "v1"


def _sorted_str_keys(values):
    return sorted({str(v) for v in values if v is not None})


def _extract_labels_from_graphs(graphs):
    node_labels = set()
    edge_types = set()
    for g in graphs:
        if not isinstance(g, (nx.Graph, nx.DiGraph)):
            continue
        for _, data in g.nodes(data=True):
            node_labels.add(str(data.get("semantic_label", data.get("label", "unknown"))))
        for _, _, data in g.edges(data=True):
            edge_type = None
            for key in ("type_str", "type", "relation", "edge_type", "label", "input_label"):
                if data.get(key) is not None:
                    edge_type = data.get(key)
                    break
            edge_types.add(str(edge_type if edge_type is not None else "unknown"))
    return _sorted_str_keys(node_labels), _sorted_str_keys(edge_types)


def _build_vocab(values):
    # Reserve id=0 for unknown/unseen values.
    vocab = {"UNK": 0}
    for i, value in enumerate(values, start=1):
        if value == "UNK":
            continue
        vocab[value] = i
    return vocab


def save_vocab(vocab_path, vocab):
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)


def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    if "UNK" not in vocab:
        vocab["UNK"] = 0
    return vocab


def initialize_or_load_vocabs(
    graphs,
    vocab_dir="artifacts/vocab",
    vocab_version=VOCAB_SCHEMA_VERSION,
    require_vocab=False,
):
    os.makedirs(vocab_dir, exist_ok=True)
    meta_path = os.path.join(vocab_dir, "metadata.json")
    node_path = os.path.join(vocab_dir, "node_label_to_id.json")
    edge_path = os.path.join(vocab_dir, "edge_type_to_id.json")

    exists = os.path.exists(node_path) and os.path.exists(edge_path) and os.path.exists(meta_path)

    if exists:
        node_vocab = load_vocab(node_path)
        edge_vocab = load_vocab(edge_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        if metadata.get("vocab_version") != vocab_version:
            raise ValueError(
                "Vocabulary version mismatch: expected {}, got {}".format(
                    vocab_version, metadata.get("vocab_version")
                )
            )
        return node_vocab, edge_vocab, metadata

    if require_vocab:
        raise FileNotFoundError(
            "Vocabulary artifacts missing in '{}', and --require_vocab was set".format(vocab_dir)
        )

    node_values, edge_values = _extract_labels_from_graphs(graphs)
    node_vocab = _build_vocab(node_values)
    edge_vocab = _build_vocab(edge_values)
    metadata = {
        "vocab_version": vocab_version,
        "num_node_labels": len(node_vocab),
        "num_edge_types": len(edge_vocab),
    }
    save_vocab(node_path, node_vocab)
    save_vocab(edge_path, edge_vocab)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return node_vocab, edge_vocab, metadata
