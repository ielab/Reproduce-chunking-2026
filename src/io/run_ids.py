import os
import json
import hashlib
import datetime
from typing import Any, Dict


def _stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _short_hash(*parts: str, n: int = 8) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:n]

def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_chunk_run_id(processor: Dict[str, Any], chunker: Dict[str, Any]) -> str:
    """
    ID format: datatime-chunker_name-processor_name-h
    """
    core = {
        "processor": processor,   # {"name": "...", **kwargs}
        "chunker": chunker,       # {"name": "...", **kwargs}
    }
    h = _short_hash(_stable_json(core))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    chunker_name = chunker.get("chunker_name", "chunker")
    # processor_name = processor.get("processor_name", "processor")

    # return f"{ts}-{chunker_name}-{h}"
    return f"{chunker_name}"


def build_emb_run_id(chunk_run_id: str,
                     encoder: Dict[str, Any],
                     method: Dict[str, Any] | None = None) -> str:

    core = {
        "chunk_run_id": chunk_run_id,
        "encoder": encoder,   # {"name":"model_encoder","backend":"openai:...","dim":1536, ...}
    }
    # print("-------------")
    # import pprint
    # pprint.pprint(encoder)

    h = _short_hash(_stable_json(core))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    enc_name = encoder.get("encoder_name")
    # backbone  = encoder.get("backbone")
    model_name = encoder.get("model_name").split("/")[-1]

    # return f"{ts}-{enc_name}-{model_name}-{h}"
    return f"{enc_name}-{model_name}"


def build_query_run_id(processor: Dict[str, Any]) -> str:

    core = {
        'processor': processor   # {"name": "...", **kwargs}
    }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    h = _short_hash(_stable_json(core))
    processor_name = processor.get("processor_name", "processor")

    return f"{ts}-{processor_name}-{h}"


def build_query_embedding_run_id(qs_id, encoder: Dict[str, Any]) -> str:

    core = {
        'qs_id': qs_id,
        'encoder': encoder
    }
    h = _short_hash(_stable_json(core))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    enc_name = encoder.get("encoder_name")
    # backbone = encoder.get("backbone")
    model_name = encoder.get("model_name").split("/")[-1]

    # return f"{ts}-{enc_name}-{model_name}-{h}"
    return f"{model_name}"



class Paths:
    def __init__(self, dataset_name, base_dir: str = "src/outputs"):
        self.base = os.path.join(base_dir, dataset_name)

    # Chunk Set
    def cs_dir(self, chunk_run_id: str) -> str:
        return os.path.join(self.base, "chunks", chunk_run_id)
    def cs_manifest(self, chunk_run_id: str) -> str:
        return os.path.join(self.cs_dir(chunk_run_id), "manifest.json")
    def cs_chunks_path(self, chunk_run_id: str) -> str:
        return os.path.join(self.cs_dir(chunk_run_id), "chunks.jsonl")

    # Embedding Run (under a chunk_run_id)
    def er_dir(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.base, "embeddings", chunk_run_id, embed_run_id)
    def er_manifest(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "manifest.json")
    def er_embeddings_jsonl(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "embeddings.jsonl")
    def er_embeddings_parquest(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "embeddings.parquet")
    def er_embeddings_gzip(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "embeddings.jsonl.gz")
    def er_embeddings_zst(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "embeddings.jsonl.zst")
    def er_npz_dir(self, chunk_run_id: str, embed_run_id: str) -> str:
        return self.er_dir(chunk_run_id, embed_run_id)
    def er_embeddings_pkl(self, chunk_run_id: str, embed_run_id: str) -> str:
        return os.path.join(self.er_dir(chunk_run_id, embed_run_id), "embeddings.pkl")

    # Query Set
    def qs_dir(self, qsid: str) -> str:
        return os.path.join(self.base, "queries", qsid)

    def qs_manifest(self, qsid: str) -> str:
        return os.path.join(self.qs_dir(qsid), "manifest.json")

    def qs_queries_path(self, qsid: str) -> str:
        return os.path.join(self.qs_dir(qsid), "queries.jsonl")

    # Query Embeddings
    def q_embed_dir(self, qsid: str, qerid: str) -> str:
        return os.path.join(self.base, "query_embeddings", qsid, qerid)
    def q_embed_manifest(self, qsid: str, qerid: str) -> str:
        return os.path.join(self.q_embed_dir(qsid, qerid), "manifest.json")
    def q_embeddings_jsonl(self, qsid: str, qerid: str) -> str:
        return os.path.join(self.q_embed_dir(qsid, qerid), "embeddings.jsonl")
    def q_embeddings_pkl(self, qsid: str, qerid: str) -> str:
        return os.path.join(self.q_embed_dir(qsid, qerid), "embeddings.pkl")


# --------------- manifest io ---------------

def write_json(path: str, obj: Dict[str, Any]):

    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def write_chunk_manifest(paths: Paths,
                         chunk_run_id: str,
                         processor: Dict[str, Any],
                         chunker: Dict[str, Any]):
    manifest = {
        "kind": "chunk_set",
        "created_at": _now_iso(),
        "chunk_run_id": chunk_run_id,
        "processor": processor,
        "chunker": chunker,
        "files": {"chunks": paths.cs_chunks_path(chunk_run_id)},
    }
    write_json(paths.cs_manifest(chunk_run_id), manifest)
    return manifest


def write_embedding_manifest(paths: Paths,
                             chunk_run_id: str,
                             embed_run_id: str,
                             *,
                             encoder: Dict[str, Any]):
    manifest = {
    "kind": "embedding_run",
    "chunk_run_id": chunk_run_id,
    "embed_run_id": embed_run_id,
    "created_at": _now_iso(),
    "encoder_init": encoder,
    "inputs": {"chunks_manifest": paths.cs_manifest(chunk_run_id)},
    "files": {"embeddings": paths.er_embeddings_jsonl(chunk_run_id, embed_run_id)},
    }
    write_json(paths.er_manifest(chunk_run_id, embed_run_id), manifest)
    return manifest


def write_query_manifest(paths: Paths,
                         query_run_id: str,
                         processor: Dict[str, Any]):
    manifest = {
        "kind": "query_set",
        "created_at": _now_iso(),
        "query_run_id": query_run_id,
        "processor": processor,
        "files": {"chunks": paths.qs_queries_path(query_run_id)},
    }
    write_json(paths.qs_manifest(query_run_id), manifest)
    return manifest


def write_query_embedding_manifest(paths: Paths,
                                   query_run_id: str,
                                   q_embed_id: str,
                                   *,
                                   encoder: Dict[str, Any]):
    man = {
        "kind": "query_embedding_run",
        "created_at": _now_iso(),
        "query_run_id": query_run_id,
        "q_embed_id": q_embed_id,
        "encoder": encoder,
        "inputs": {"queries_manifest": paths.q_embed_manifest(query_run_id, q_embed_id)},
        "files": {"embeddings": paths.q_embeddings_jsonl(query_run_id, q_embed_id)},
    }

    write_json(paths.q_embed_manifest(query_run_id, q_embed_id), man); return man