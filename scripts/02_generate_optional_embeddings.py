import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _read_sequences(feather_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_feather(feather_path)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(f"Expected columns id, sequence in {feather_path}; got {list(df.columns)}")
    ids = df["id"].astype(str).tolist()
    seqs = df["sequence"].astype(str).tolist()
    return ids, seqs


def _smart_order(seqs: list[str]) -> np.ndarray:
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64)
    return np.argsort(lengths)[::-1]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: (B, L, D), attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    x = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return x.sum(dim=1) / denom


@torch.no_grad()
def embed_esm2(
    seqs: list[str],
    model_name: str,
    batch_size: int,
    max_len: int,
    device: torch.device,
) -> np.ndarray:
    from transformers import EsmModel, EsmTokenizer

    tok = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    order = _smart_order(seqs)
    seqs_sorted = [seqs[i] for i in order]

    outs: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for i in range(0, len(seqs_sorted), batch_size):
        batch = seqs_sorted[i : i + batch_size]
        ids = tok.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        ids = {k: v.to(device) for k, v in ids.items()}

        ctx = torch.amp.autocast("cuda") if use_amp else torch.autocast("cpu", enabled=False)
        with ctx:
            out = model(**ids)
        pooled = _mean_pool(out.last_hidden_state.float(), ids["attention_mask"])
        outs.append(pooled.cpu().numpy())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    embs_sorted = np.vstack(outs).astype(np.float32)
    embs = np.zeros_like(embs_sorted)
    embs[order] = embs_sorted
    return embs


@torch.no_grad()
def embed_ankh(
    seqs: list[str],
    model_name: str,
    batch_size: int,
    max_len: int,
    device: torch.device,
    trust_remote_code: bool,
) -> np.ndarray:
    # Ankh models are HuggingFace-hosted and may require trust_remote_code.
    # We treat them as encoder-style models and mean-pool hidden states over attention_mask.
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
    model.eval()

    order = _smart_order(seqs)
    seqs_sorted = [seqs[i] for i in order]

    outs: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for i in range(0, len(seqs_sorted), batch_size):
        batch = seqs_sorted[i : i + batch_size]

        # Many protein LMs expect space-separated amino acids; if the tokenizer has a small vocab,
        # this typically helps. If it hurts, you can disable via --ankh-space-sep 0.
        batch = [" ".join(list(s.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X"))) for s in batch]

        ids = tok(
            batch,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        ids = {k: v.to(device) for k, v in ids.items()}

        ctx = torch.amp.autocast("cuda") if use_amp else torch.autocast("cpu", enabled=False)
        with ctx:
            out = model(**ids)

        # HuggingFace convention: encoder outputs `last_hidden_state`
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            raise RuntimeError(f"Model {model_name} did not return last_hidden_state")

        pooled = _mean_pool(last.float(), ids["attention_mask"])
        outs.append(pooled.cpu().numpy())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    embs_sorted = np.vstack(outs).astype(np.float32)
    embs = np.zeros_like(embs_sorted)
    embs[order] = embs_sorted
    return embs


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate optional multimodal embeddings as .npy artefacts.")
    ap.add_argument(
        "--artefacts-dir",
        type=Path,
        default=Path("artefacts_local") / "artefacts",
        help="Path containing parsed/ and features/ (default: artefacts_local/artefacts)",
    )
    ap.add_argument(
        "--mode",
        choices=["esm2_3b", "ankh", "text"],
        required=True,
        help="Which optional embedding to generate.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda|cpu (default: cuda)",
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)

    ap.add_argument(
        "--esm2-3b-model",
        type=str,
        default="facebook/esm2_t36_3B_UR50D",
        help="HF model id for ESM2-3B",
    )
    ap.add_argument(
        "--ankh-model",
        type=str,
        default="ElnaggarLab/ankh-large",
        help="HF model id for Ankh (large is typically 1536D)",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True for Ankh loading (often required).",
    )

    # Text mode (fixed-width sparse->dense features; default dimension = 10279)
    ap.add_argument(
        "--text-path",
        type=Path,
        default=None,
        help="Path to a 2-column TSV/CSV with EntryID and text (required for --mode text).",
    )
    ap.add_argument(
        "--text-sep",
        type=str,
        default="\t",
        help="Separator for --text-path (default: tab).",
    )
    ap.add_argument(
        "--text-id-col",
        type=str,
        default="EntryID",
        help="ID column name in --text-path (default: EntryID).",
    )
    ap.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Text column name in --text-path (default: text).",
    )
    ap.add_argument(
        "--text-dim",
        type=int,
        default=10279,
        help="Output feature dimension for text TF-IDF (default: 10279).",
    )
    ap.add_argument(
        "--text-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="On-disk dtype for text .npy (default: float16 to keep size sane).",
    )
    ap.add_argument(
        "--text-ngram-max",
        type=int,
        default=2,
        help="Max n-gram for TF-IDF (default: 2).",
    )

    args = ap.parse_args()

    artefacts_dir: Path = args.artefacts_dir
    parsed_dir = artefacts_dir / "parsed"
    feat_dir = artefacts_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    train_feather = parsed_dir / "train_seq.feather"
    test_feather = parsed_dir / "test_seq.feather"
    if not train_feather.exists() or not test_feather.exists():
        raise FileNotFoundError(
            f"Expected {train_feather} and {test_feather}. Run Phase 1 parsing first."
        )

    # Make HF caches relocatable (handy on Colab/Kaggle)
    os.environ.setdefault("HF_HOME", str(artefacts_dir / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(artefacts_dir / "hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(artefacts_dir / "torch_cache"))

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    print("Loading sequences...")
    train_ids, train_seqs = _read_sequences(train_feather)
    test_ids, test_seqs = _read_sequences(test_feather)

    if args.mode == "esm2_3b":
        print(f"Embedding ESM2-3B: {args.esm2_3b_model}")
        train_emb = embed_esm2(train_seqs, args.esm2_3b_model, args.batch_size, args.max_len, device)
        test_emb = embed_esm2(test_seqs, args.esm2_3b_model, args.batch_size, args.max_len, device)
        np.save(feat_dir / "train_embeds_esm2_3b.npy", train_emb)
        np.save(feat_dir / "test_embeds_esm2_3b.npy", test_emb)
        print(f"Saved: {feat_dir / 'train_embeds_esm2_3b.npy'}")
        print(f"Saved: {feat_dir / 'test_embeds_esm2_3b.npy'}")
        return 0

    if args.mode == "ankh":
        print(f"Embedding Ankh: {args.ankh_model}")
        train_emb = embed_ankh(
            train_seqs,
            args.ankh_model,
            args.batch_size,
            args.max_len,
            device,
            trust_remote_code=args.trust_remote_code,
        )
        test_emb = embed_ankh(
            test_seqs,
            args.ankh_model,
            args.batch_size,
            args.max_len,
            device,
            trust_remote_code=args.trust_remote_code,
        )
        np.save(feat_dir / "train_embeds_ankh.npy", train_emb)
        np.save(feat_dir / "test_embeds_ankh.npy", test_emb)
        print(f"Saved: {feat_dir / 'train_embeds_ankh.npy'}")
        print(f"Saved: {feat_dir / 'test_embeds_ankh.npy'}")
        return 0

    if args.mode == "text":
        if args.text_path is None:
            raise ValueError("--text-path is required for --mode text")

        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        if not args.text_path.exists():
            raise FileNotFoundError(f"text-path not found: {args.text_path}")

        print(f"Loading text corpus: {args.text_path}")
        df = pd.read_csv(args.text_path, sep=args.text_sep, dtype=str)
        if args.text_id_col not in df.columns or args.text_col not in df.columns:
            raise ValueError(
                f"Expected columns {args.text_id_col!r}, {args.text_col!r} in {args.text_path}; got {list(df.columns)}"
            )

        # Many-to-one mapping is possible (multiple pubs per EntryID). Concatenate.
        df = df[[args.text_id_col, args.text_col]].dropna()
        df[args.text_id_col] = df[args.text_id_col].astype(str)
        df[args.text_col] = df[args.text_col].astype(str)
        grouped = df.groupby(args.text_id_col, sort=False)[args.text_col].apply(lambda x: " \n ".join(x.tolist()))
        text_map = grouped.to_dict()

        def _norm_uniprot_id(pid: str) -> str:
            # Common UniProt FASTA headers: sp|P12345|NAME_HUMAN ...
            parts = str(pid).split("|")
            if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
                return parts[1]
            if len(parts) >= 3:
                return parts[1]
            return str(pid)

        # Support both raw IDs and normalised accessions for joining.
        text_map_norm: dict[str, str] = {}
        for k, v in text_map.items():
            text_map_norm[str(k)] = v
            text_map_norm[_norm_uniprot_id(k)] = v

        train_texts = [text_map_norm.get(str(pid), "") for pid in train_ids]
        test_texts = [text_map_norm.get(str(pid), "") for pid in test_ids]

        # TF-IDF gives a fixed-width, high-dimensional text modality with controllable size.
        # This is the most practical way to realise a 10279D "text embedding" without a bespoke LLM pipeline.
        print(f"Fitting TF-IDF (dim={args.text_dim}, ngram<= {args.text_ngram_max})...")
        vec = TfidfVectorizer(
            max_features=args.text_dim,
            ngram_range=(1, max(1, int(args.text_ngram_max))),
            min_df=2,
            strip_accents="unicode",
            lowercase=True,
        )
        vec.fit(train_texts + test_texts)

        n_features = len(vec.get_feature_names_out())
        print(f"TF-IDF vocab size: {n_features} (padded to {args.text_dim})")

        # Persist vectorizer for reproducibility
        joblib.dump(vec, feat_dir / "text_vectorizer.joblib")
        print(f"Saved: {feat_dir / 'text_vectorizer.joblib'}")

        out_dtype = np.float16 if args.text_dtype == "float16" else np.float32

        def _write_memmap(name: str, texts: list[str]):
            path = feat_dir / name
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=out_dtype, shape=(len(texts), args.text_dim))

            bs = 2048
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                X = vec.transform(chunk)  # sparse
                # Convert per-chunk to dense to write to .npy
                arr = X.toarray().astype(out_dtype, copy=False)
                if arr.shape[1] == args.text_dim:
                    mm[i : i + arr.shape[0], :] = arr
                else:
                    dense = np.zeros((arr.shape[0], args.text_dim), dtype=out_dtype)
                    if arr.shape[1] > 0:
                        dense[:, : arr.shape[1]] = arr
                    mm[i : i + dense.shape[0], :] = dense
            mm.flush()
            return path

        train_path = _write_memmap("train_embeds_text.npy", train_texts)
        test_path = _write_memmap("test_embeds_text.npy", test_texts)
        print(f"Saved: {train_path}")
        print(f"Saved: {test_path}")
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
