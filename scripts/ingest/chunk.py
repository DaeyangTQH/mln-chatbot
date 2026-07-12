"""Create structure-aware chunks with parent pointers for small-to-big retrieval."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import tiktoken

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config


try:
    ENCODING = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")


UPPER_VI = "A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
SENTENCE_RE = re.compile(rf"(?<=[.!?…])\s+(?=(?:[{UPPER_VI}0-9\"“]|[a-zđ]\)))")


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected = (
        text.replace("V.I.", "V_I_")
        .replace("C. Mác", "C_Mác")
        .replace("Ph. Ăngghen", "Ph_Ăngghen")
        .replace("Sđd.", "Sđd_")
        .replace("tr.", "tr_")
        .replace("t.", "t_")
    )
    parts = [part.strip() for part in SENTENCE_RE.split(protected) if part.strip()]
    return [
        part.replace("V_I_", "V.I.")
        .replace("C_Mác", "C. Mác")
        .replace("Ph_Ăngghen", "Ph. Ăngghen")
        .replace("Sđd_", "Sđd.")
        .replace("tr_", "tr.")
        .replace("t_", "t.")
        for part in parts
    ]


def chunk_body(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_count(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            overlap: list[str] = []
            overlap_count = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_count += token_count(previous)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = overlap_count

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def infer_topic(node: dict[str, Any], text: str) -> str | None:
    haystack = f"{node.get('source_title') or ''} {node.get('breadcrumb') or ''} {text}".lower()
    body = text.lower()
    if "độc quyền là sự liên minh" in body:
        return "khai_niem_doc_quyen"
    if "độc quyền nhà nước là kiểu độc quyền" in body:
        return "doc_quyen_nha_nuoc"
    if "điện lực" in haystack:
        return "dien_doc_quyen_nha_nuoc"
    if "nguyên nhân hình thành độc quyền và độc quyền nhà nước" in haystack:
        return "nguyen_nhan_doc_quyen_va_doc_quyen_nha_nuoc"
    if "nguyên nhân hình thành độc quyền" in haystack:
        return "nguyen_nhan_doc_quyen"
    if "độc quyền nhà nước" in haystack and "ngành điện" in haystack:
        return "dien_doc_quyen_nha_nuoc"
    if "cấp nước" in haystack or "giá nước sạch" in haystack:
        return "nuoc_sach_dieu_tiet_nha_nuoc"
    if "độc quyền nhà nước" in haystack:
        return "doc_quyen_nha_nuoc"
    if "độc quyền là" in haystack:
        return "khai_niem_doc_quyen"
    if "khuyết tật của kinh tế thị trường" in haystack or "khuyết tật của thị trường" in haystack:
        return "khuyet_tat_thi_truong"
    if "doanh nghiệp nhà nước chỉ tập trung vào các lĩnh vực then chốt" in haystack:
        return "viet_nam_linh_vuc_then_chot"
    return None


def topic_label(topic: str | None) -> str:
    labels = {
        "khai_niem_doc_quyen": "khái niệm độc quyền",
        "nguyen_nhan_doc_quyen": "nguyên nhân hình thành độc quyền",
        "nguyen_nhan_doc_quyen_va_doc_quyen_nha_nuoc": "nguyên nhân hình thành độc quyền và độc quyền nhà nước",
        "doc_quyen_nha_nuoc": "độc quyền nhà nước",
        "dien_doc_quyen_nha_nuoc": "ngành điện, độc quyền nhà nước, điều tiết Nhà nước",
        "nuoc_sach_dieu_tiet_nha_nuoc": "ngành nước sạch, điều tiết Nhà nước",
        "khuyet_tat_thi_truong": "khuyết tật thị trường",
        "viet_nam_linh_vuc_then_chot": "Việt Nam, lĩnh vực then chốt, doanh nghiệp nhà nước",
    }
    return labels.get(topic or "", "")


def parent_record(node: dict[str, Any], parent_id: str) -> dict[str, Any]:
    breadcrumb = node.get("breadcrumb") or node.get("section_path") or "Nguồn"
    text = f"{breadcrumb}\n\n{node['text']}".strip()
    return {
        "id": parent_id,
        "source_type": node.get("source_type"),
        "source_title": node.get("source_title"),
        "source_file": node.get("source_file"),
        "source_url": node.get("source_url"),
        "citation": node.get("citation"),
        "chapter": node.get("chapter"),
        "section_path": node.get("section_path"),
        "page_start": node.get("page_start"),
        "page_end": node.get("page_end"),
        "token_count": token_count(text),
        "text": text,
    }


def build_chunks(structured_path: Path = config.STRUCTURED_PATH) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    parents: dict[str, Any] = {}

    for node in structured["nodes"]:
        parent_id = f"parent_{node['id']}"
        parents[parent_id] = parent_record(node, parent_id)
        breadcrumb = node.get("breadcrumb") or node.get("section_path") or "Nguồn"
        body_chunks = chunk_body(
            node["text"],
            max_tokens=config.CHUNK_MAX_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
        )
        if not body_chunks:
            continue

        for chunk_idx, body in enumerate(body_chunks, start=1):
            chunk_id = f"{node['id']}_chunk_{chunk_idx:02d}"
            topic = infer_topic(node, body)
            topic_text = topic_label(topic)
            embedding_text = f"{breadcrumb}\n{('Chủ đề: ' + topic_text) if topic_text else ''}\n\n{body}".strip()
            chunks.append(
                {
                    "id": chunk_id,
                    "parent_id": parent_id,
                    "source_type": node.get("source_type"),
                    "source_title": node.get("source_title"),
                    "source_file": node.get("source_file"),
                    "source_url": node.get("source_url"),
                    "citation": node.get("citation"),
                    "chapter": node.get("chapter"),
                    "section_path": node.get("section_path"),
                    "page_start": node.get("page_start"),
                    "page_end": node.get("page_end"),
                    "topic": topic,
                    "token_count": token_count(body),
                    "text": body,
                    "embedding_text": embedding_text,
                }
            )

    return chunks, parents


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", type=Path, default=config.STRUCTURED_PATH)
    args = parser.parse_args()

    config.ensure_output_dir()
    chunks, parents = build_chunks(args.structured)
    write_jsonl(config.CHUNKS_PATH, chunks)
    config.PARENTS_PATH.write_text(json.dumps(parents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = {
        "chunks_path": str(config.CHUNKS_PATH),
        "parents_path": str(config.PARENTS_PATH),
        "chunks": len(chunks),
        "parents": len(parents),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
