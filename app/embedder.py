"""Query embeddings for the MCP's vector search (`POST /embed`).

The embedding configuration is NOT written here: it is read from the corpus manifest (corpus/manifest.json,
produced by the corpus build in the `train` repo). That guarantees /embed uses exactly the model, revision,
normalization, max length and query prefix the corpus was built with. If the manifest is missing or invalid,
the service refuses to start.
"""
import json
import os
from pathlib import Path

from app.embedding import CONTRACT_FIELDS, Embedder, EmbeddingConfig

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "corpus" / "manifest.json"
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", DEFAULT_MANIFEST))

_embedder: Embedder | None = None
_info: dict = {}


def load_config(path: Path = MANIFEST_PATH) -> tuple[EmbeddingConfig, dict]:
    if not path.exists():
        raise RuntimeError(
            f"Falta el manifest del corpus: {path}. Copia train/artifacts/corpus/manifest.json a engine/corpus/ "
            f"o define MANIFEST_PATH."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    emb = manifest.get("embedding")
    missing = [f for f in CONTRACT_FIELDS if not isinstance(emb, dict) or f not in emb]
    if missing:
        raise RuntimeError(f"El manifest {path} no tiene la configuración de embedding completa (faltan {missing})")
    # If corpus.yml did not pin `revision`, use the revision the corpus was actually built with.
    revision = emb["revision"] or emb.get("resolved_revision")
    config = EmbeddingConfig.from_dict({**emb, "revision": revision})
    return config, manifest


def load(path: Path = MANIFEST_PATH) -> None:
    """Load the embedding model once, at startup."""
    global _embedder, _info
    config, manifest = load_config(path)
    _embedder = Embedder(config)  # loads the model (from the Hugging Face cache, or downloads it)
    _info = {
        "model_id": config.model_id,
        "revision": config.revision,
        "dimension": config.dimension,
        "corpus_built_at": manifest.get("built_at"),
    }


def info() -> dict:
    return dict(_info)


def embed(text: str) -> list[float]:
    if _embedder is None:
        raise RuntimeError("embedding model not loaded")
    return _embedder.embed_query(text).tolist()  # the query prefix is applied inside embed_query
