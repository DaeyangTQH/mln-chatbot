"""Chạy bộ test văn phong/hành vi qua API /api/chat và xuất file QA để duyệt tay.

Nạp từng case trong eval/style_cases.jsonl vào endpoint /api/chat (mô hình
GENERATION_MODEL trả lời), gom luồng SSE thành câu trả lời, gắn vài cờ heuristic
(KHÔNG dùng LLM chấm điểm), rồi ghi ra output/eval/style_qa_<timestamp>.md + .json.

Không chấm đúng/sai nội dung — người dùng tự đọc file .md và tự đánh giá.

Ví dụ:
    uvicorn scripts.api:app --port 8899        # bật server ở terminal khác trước
    python scripts/eval_style.py               # chạy toàn bộ
    python scripts/eval_style.py --limit 5      # chạy thử 5 case
    python scripts/eval_style.py --filter prompt_injection
    python scripts/eval_style.py --base-url https://<railway-app>.up.railway.app
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_CASES = ROOT_DIR / "eval" / "style_cases.jsonl"
DEFAULT_OUT_DIR = ROOT_DIR / "output" / "eval"

# Cụm từ gợi ý bot đang "từ chối / báo ngoài phạm vi" (heuristic, có dấu lẫn không dấu).
_REFUSAL_PATTERNS = [
    "ngoài corpus",
    "ngoài phạm vi",
    "không có trong corpus",
    "không nằm trong",
    "không đề cập",
    "không có thông tin",
    "chưa có trong",
    "không thuộc phạm vi",
    "không tìm thấy",
    "không cung cấp",
    "vui lòng",
    "bạn có thể nói rõ",
    "làm rõ",
]
# Dấu tiếng Việt để đoán câu trả lời có phải tiếng Việt không.
_VI_DIACRITICS = "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Không thấy file cases: {path}")
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON hỏng ở dòng {line_no} của {path}: {exc}")
    return cases


def call_chat(base_url: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Gọi /api/chat, gom các event SSE 'token' thành 1 chuỗi. Trả về answer + lỗi (nếu có)."""

    payload = {
        "question": case["question"],
        "history": case.get("history") or [],
        "mode": case.get("mode") or "default",
        "pinned": case.get("pinned"),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    tokens: list[str] = []
    stream_error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line.startswith("data:"):
                    continue  # bỏ qua dòng "event:" và dòng trống
                body = line[len("data:") :].strip()
                if not body:
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "token":
                    tokens.append(event.get("token", ""))
                elif etype == "error":
                    stream_error = event.get("message", "unknown SSE error")
    except urllib.error.HTTPError as exc:
        return {"answer": "", "error": f"HTTP {exc.code}: {exc.reason}"}
    except urllib.error.URLError as exc:
        return {"answer": "", "error": f"Không kết nối được ({exc.reason}). Server đã bật chưa?"}
    except TimeoutError:
        return {"answer": "", "error": f"Timeout sau {timeout}s"}

    return {"answer": "".join(tokens), "error": stream_error}


def compute_flags(answer: str, error: str | None) -> dict[str, Any]:
    """Cờ heuristic hỗ trợ mắt người duyệt — KHÔNG phải phán quyết đúng/sai."""

    lowered = answer.lower()
    return {
        "error": error,
        "empty": not answer.strip(),
        "len": len(answer),
        "markdown_leak": bool(re.search(r"\*\*|^#{1,6}\s|`|\|", answer, flags=re.MULTILINE)),
        "has_nguon": "nguồn" in lowered,
        "page_cite": bool(re.search(r"tr\.\s*\d+", lowered)),
        "refusal_signal": any(p in lowered for p in _REFUSAL_PATTERNS),
        "non_vietnamese": bool(answer.strip()) and not any(ch in _VI_DIACRITICS for ch in lowered),
    }


def _flag_str(flags: dict[str, Any]) -> str:
    def yn(value: bool) -> str:
        return "yes" if value else "no"

    parts = [
        f"markdown_leak={yn(flags['markdown_leak'])}",
        f"has_nguon={yn(flags['has_nguon'])}",
        f"page_cite={yn(flags['page_cite'])}",
        f"refusal={yn(flags['refusal_signal'])}",
        f"non_vi={yn(flags['non_vietnamese'])}",
        f"len={flags['len']}",
    ]
    if flags["error"]:
        parts.append(f"ERROR={flags['error']}")
    if flags["empty"]:
        parts.append("EMPTY")
    return " · ".join(parts)


def write_markdown(path: Path, results: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Kết quả test văn phong/hành vi chatbot")
    lines.append("")
    lines.append(f"- Thời điểm: {meta['timestamp']}")
    lines.append(f"- API: {meta['base_url']}")
    lines.append(f"- Số case: {meta['total']}")
    lines.append(f"- Lỗi/không trả lời được: {meta['errors']}")
    lines.append("")
    lines.append("> Các cờ chỉ là gợi ý heuristic (không phải chấm đúng/sai). "
                 "Hãy tự đọc phần A của mỗi case và đối chiếu 'Bẫy' / 'Fail nếu'.")
    lines.append("")

    lines.append("## Tổng hợp theo nhóm")
    lines.append("")
    lines.append("| Nhóm | Số case | markdown_leak | non_vietnamese | error |")
    lines.append("|---|---|---|---|---|")
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rows in by_cat.items():
        md_leak = sum(1 for r in rows if r["flags"]["markdown_leak"])
        non_vi = sum(1 for r in rows if r["flags"]["non_vietnamese"])
        errs = sum(1 for r in rows if r["flags"]["error"] or r["flags"]["empty"])
        lines.append(f"| {cat} | {len(rows)} | {md_leak} | {non_vi} | {errs} |")
    lines.append("")

    lines.append("## Chi tiết từng case")
    lines.append("")
    for r in results:
        lines.append(
            f"### [{r['id']}] {r['category']} · {r['difficulty']} · expect={r['expected_behavior']}"
        )
        lines.append(f"**Q:** {r['question']}")
        if r.get("pinned"):
            lines.append(f"**Pinned:** {r['pinned']}")
        if r.get("history"):
            hist = " | ".join(f"{m['role']}: {m['content']}" for m in r["history"])
            lines.append(f"**History:** {hist}")
        lines.append(f"**Bẫy:** {r.get('trap', '')}")
        lines.append(f"**Fail nếu:** {r.get('fail_if', '')}")
        lines.append("")
        lines.append("**A:**")
        lines.append("")
        answer = r["answer"].strip() or "(không có nội dung)"
        for para in answer.split("\n"):
            lines.append(f"> {para}" if para.strip() else ">")
        lines.append("")
        lines.append(f"**Cờ:** {_flag_str(r['flags'])}")
        lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bộ test văn phong qua /api/chat và xuất file QA.")
    parser.add_argument("--base-url", default="http://localhost:8899", help="Gốc API (mặc định local).")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="File .jsonl chứa case.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Thư mục xuất file.")
    parser.add_argument("--filter", dest="category", default=None, help="Chỉ chạy 1 nhóm category.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số case (chạy thử).")
    parser.add_argument("--delay", type=float, default=0.5, help="Giãn cách giữa các case (giây).")
    parser.add_argument("--mode", default=None, help="Ép mode cho mọi case (default|socratic|debate).")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout mỗi request (giây).")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("Không có case nào khớp bộ lọc.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: list[dict[str, Any]] = []
    print(f"Chạy {len(cases)} case qua {args.base_url} ...")
    for idx, case in enumerate(cases, start=1):
        if args.mode:
            case = {**case, "mode": args.mode}
        print(f"  [{idx}/{len(cases)}] {case['id']} ({case.get('category')})", flush=True)
        outcome = call_chat(args.base_url, case, timeout=args.timeout)
        flags = compute_flags(outcome["answer"], outcome["error"])
        results.append({**case, "answer": outcome["answer"], "flags": flags})
        if idx < len(cases) and args.delay > 0:
            time.sleep(args.delay)

    errors = sum(1 for r in results if r["flags"]["error"] or r["flags"]["empty"])
    meta = {
        "timestamp": timestamp,
        "base_url": args.base_url,
        "total": len(results),
        "errors": errors,
        "flag_summary": dict(Counter(
            k for r in results for k, v in r["flags"].items()
            if k in {"markdown_leak", "non_vietnamese", "has_nguon", "page_cite", "refusal_signal"} and v
        )),
    }

    md_path = args.out_dir / f"style_qa_{timestamp}.md"
    json_path = args.out_dir / f"style_qa_{timestamp}.json"
    write_markdown(md_path, results, meta)
    json_path.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nXong. {errors}/{len(results)} case lỗi/không trả lời được.")
    print(f"  Markdown (duyệt tay): {md_path}")
    print(f"  JSON (máy đọc):       {json_path}")


if __name__ == "__main__":
    main()
