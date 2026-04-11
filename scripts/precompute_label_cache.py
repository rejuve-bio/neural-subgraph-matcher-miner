#!/usr/bin/env python3
"""Precompute frozen text-label embeddings for semantic mining.
"""

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
from tqdm import tqdm

from common.label_encoder import UniversalLabelEncoder


TEXT_EDGE_KEYS = ("type_str", "edge_type", "relation", "label", "input_label", "type")
NODE_LABEL_KEYS = ("semantic_label", "label", "kind", "group", "type")
SCRIPT_SEMANTIC_PRESETS = {
    "biology": {
        "node_labels": ["TF", "Gene", "Enzyme", "mRNA", "Metabolite"],
        "edge_types": ["regulates", "transcribes", "translates", "catalyzes"],
    },
    "ecommerce": {
        "node_labels": ["product", "category", "brand", "customer", "review"],
        "edge_types": ["viewed", "purchased", "belongs_to", "made_by", "wrote"],
    },
    "social": {
        "node_labels": ["influencer", "community", "user", "creator", "new_user", "bot"],
        "edge_types": ["follows", "mentions", "replies_to", "belongs_to"],
    },
    "universal": {
        "node_labels": [
            "Hub Entity", "Gene", "User", "Product", "Organization", "Protein",
            "Category", "Compound", "Topic", "Entity", "Document", "Disease",
            "Book", "Music", "Pathway", "Customer", "Creator", "Anatomy",
            "Location", "Event", "Leaf Entity", "Review", "Side Effect",
            "Symptom", "Metabolite", "Tag", "Comment", "Image", "Article",
            "Video", "Record",
        ],
        "edge_types": ["relation_{:02d}".format(i) for i in range(1, 64)],
    },
}


def safe_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def non_numeric_text(value):
    text = safe_text(value)
    if text is None:
        return None
    try:
        float(text)
        return None
    except ValueError:
        return text


def node_label(attrs):
    for key in NODE_LABEL_KEYS:
        text = safe_text(attrs.get(key))
        if text is not None:
            return text
    return None


def edge_label(attrs):
    for key in TEXT_EDGE_KEYS:
        text = non_numeric_text(attrs.get(key))
        if text is not None:
            return text
    return None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_from_networkx(graph, labels, include_nodes=True, include_edges=True):
    if include_nodes:
        for _, attrs in graph.nodes(data=True):
            text = node_label(attrs)
            if text is not None:
                labels.add(text)
    if include_edges:
        for _, _, attrs in graph.edges(data=True):
            text = edge_label(attrs)
            if text is not None:
                labels.add(text)


def collect_from_dict(data, labels, include_nodes=True, include_edges=True):
    if include_nodes:
        for node_record in data.get("nodes", []):
            attrs = node_record[1] if isinstance(node_record, (list, tuple)) and len(node_record) >= 2 else {}
            if isinstance(attrs, dict):
                text = node_label(attrs)
                if text is not None:
                    labels.add(text)
    if include_edges:
        for edge_record in data.get("edges", []):
            attrs = edge_record[2] if isinstance(edge_record, (list, tuple)) and len(edge_record) >= 3 else {}
            if isinstance(attrs, dict):
                text = edge_label(attrs)
                if text is not None:
                    labels.add(text)


def collect_from_graph_pickle(path, labels, include_nodes=True, include_edges=True):
    data = load_pickle(path)
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        collect_from_networkx(data, labels, include_nodes, include_edges)
    elif isinstance(data, dict):
        collect_from_dict(data, labels, include_nodes, include_edges)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (nx.Graph, nx.DiGraph)):
                collect_from_networkx(item, labels, include_nodes, include_edges)
            elif isinstance(item, dict):
                collect_from_dict(item, labels, include_nodes, include_edges)
    else:
        raise ValueError("Unsupported pickle format in {}: {}".format(path, type(data)))


def collect_from_preset(preset_name, labels, include_nodes=True, include_edges=True):
    preset = SCRIPT_SEMANTIC_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            "Unknown semantic preset '{}'. Available: {}".format(
                preset_name, ", ".join(sorted(SCRIPT_SEMANTIC_PRESETS))
            )
        )
    if include_nodes:
        for label in preset["node_labels"]:
            text = safe_text(label)
            if text is not None:
                labels.add(text)
    if include_edges:
        for label in preset["edge_types"]:
            text = safe_text(label)
            if text is not None:
                labels.add(text)


def collect_from_text_file(path, labels):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = safe_text(line)
            if text is not None and not text.startswith("#"):
                labels.add(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute cached UniversalLabelEncoder .npy vectors for node/edge labels."
    )
    parser.add_argument(
        "--graph_pkl",
        "--graph-pkl",
        action="append",
        default=[],
        help="Graph PKL to scan. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--semantic_preset",
        "--semantic-preset",
        action="append",
        default=[],
        help="Synthetic semantic preset to cache, e.g. universal. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--labels_file",
        "--labels-file",
        action="append",
        default=[],
        help="Text file containing one label per line.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Literal label text to cache. Repeat for multiple labels.",
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        required=True,
        help="Directory where .npy label embeddings are stored.",
    )
    parser.add_argument(
        "--backend",
        default="sentence_transformers",
        choices=["sentence_transformers", "cache_only", "hashing", "auto"],
        help="Use sentence_transformers to create MiniLM cache; use cache_only to verify coverage.",
    )
    parser.add_argument(
        "--label_encoder_name",
        "--label-encoder-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nodes_only", action="store_true", help="Only collect node labels from graph PKLs.")
    parser.add_argument("--edges_only", action="store_true", help="Only collect edge labels from graph PKLs.")
    parser.add_argument("--dry_run", action="store_true", help="Print labels but do not encode.")
    args = parser.parse_args()
    if args.nodes_only and args.edges_only:
        parser.error("--nodes_only and --edges_only are mutually exclusive")
    return args


def expanded_csv(items):
    out = []
    for item in items:
        out.extend(x.strip() for x in item.split(",") if x.strip())
    return out


def main():
    args = parse_args()
    include_nodes = not args.edges_only
    include_edges = not args.nodes_only

    labels = set()

    for preset_name in expanded_csv(args.semantic_preset):
        print("Collecting labels from semantic preset: {}".format(preset_name), file=sys.stderr)
        collect_from_preset(preset_name, labels, include_nodes, include_edges)

    for graph_path in args.graph_pkl:
        print("Collecting labels from graph PKL: {}".format(graph_path), file=sys.stderr)
        collect_from_graph_pickle(graph_path, labels, include_nodes, include_edges)

    for labels_file in args.labels_file:
        print("Collecting labels from text file: {}".format(labels_file), file=sys.stderr)
        collect_from_text_file(labels_file, labels)

    for label in args.label:
        text = safe_text(label)
        if text is not None:
            labels.add(text)

    labels = sorted(labels)
    print("LABEL_COUNT:", len(labels))
    for label in labels:
        print(label)

    if args.dry_run:
        print("DRY_RUN_OK")
        return

    encoder = UniversalLabelEncoder(
        model_name=args.label_encoder_name,
        cache_dir=args.cache_dir,
        backend=args.backend,
        embedding_dim=args.embedding_dim,
    )

    for start in tqdm(range(0, len(labels), args.batch_size), desc="Encoding labels"):
        encoder.encode_many(labels[start:start + args.batch_size])

    verifier = UniversalLabelEncoder(
        model_name=args.label_encoder_name,
        cache_dir=args.cache_dir,
        backend="cache_only",
        embedding_dim=args.embedding_dim,
    )
    verifier.encode_many(labels)

    print("CACHE_OK:", args.cache_dir)
    print("CACHE_LABEL_COUNT:", len(labels))


if __name__ == "__main__":
    main()
