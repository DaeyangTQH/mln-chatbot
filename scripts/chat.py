"""CLI chatbot for the monopoly RAG corpus."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config
from scripts.retrieve import format_source, hybrid_retrieve


MODE_INSTRUCTIONS = {
    "default": (
        "TRẢ LỜI NHANH: đi thẳng vào trọng tâm, trình bày gọn trong tối đa 3-4 ý chính, "
        "không lan man, chốt kết luận rõ ràng. Bám sát ngữ cảnh."
    ),
    "socratic": (
        "SOCRATIC — BẮT BUỘC làm đúng: KHÔNG đưa câu trả lời đầy đủ ngay từ đầu. "
        "Mở đầu bằng 2-3 CÂU HỎI gợi mở dẫn dắt người học tự suy nghĩ (đánh số), mỗi câu kèm "
        "một gợi ý hướng đi ngắn. Chỉ chốt lại 1-2 câu kết luận cô đọng ở CUỐI. "
        "Giữ giọng khơi gợi, không giảng giải một chiều. Vẫn bám sát ngữ cảnh và trích nguồn."
    ),
    "debate": (
        "PHẢN BIỆN HỌC THUẬT — BẮT BUỘC trình bày theo đúng 4 mục sau: "
        "Luận điểm ủng hộ, Điểm yếu & phản biện, Điều kiện áp dụng, Rủi ro hiểu sai. "
        "Nêu cân bằng cả hai chiều, không thiên vị. Vẫn bám sát ngữ cảnh và trích nguồn."
    ),
}

# Nhiệt độ theo mode: default ổn định; socratic/debate cần biến thiên để khác biệt rõ.
MODE_TEMPERATURE = {
    "default": 0.2,
    "socratic": 0.6,
    "debate": 0.5,
}


def temperature_for_mode(mode: str = "default") -> float:
    return MODE_TEMPERATURE.get(mode, MODE_TEMPERATURE["default"])


def extract_guardrails() -> str:
    if not config.SYNTHESIS_PATH.exists():
        return (
            "- Không nói tuyệt đối rằng Việt Nam không tư nhân hóa điện/nước; nói không tư nhân hóa toàn bộ.\n"
            "- Phân biệt độc quyền tư nhân, độc quyền nhà nước và độc quyền tự nhiên.\n"
            "- Trả lời tiếng Việt và trích nguồn."
        )
    text = config.SYNTHESIS_PATH.read_text(encoding="utf-8")
    match = re.search(r"## 8\. Guardrails cho chatbot\s*(.+)$", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text[-1600:]


def context_block(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, context in enumerate(contexts, start=1):
        source = format_source(context)
        blocks.append(
            "\n".join(
                [
                    f"[Nguồn {idx}] {source}",
                    f"source_type={context.get('source_type')} score={context.get('score'):.4f}",
                    context["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def system_prompt_for_mode(mode: str = "default") -> str:
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["default"])
    guardrails = extract_guardrails()
    return f"""Bạn là chatbot hỗ trợ học môn Kinh tế chính trị Mác - Lênin.

Chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu câu hỏi vượt ngoài corpus, nói rõ phần nào chưa có trong corpus.
Trả lời bằng tiếng Việt, có cấu trúc vừa đủ, không bịa nguồn.
Chỉ xuất văn bản thuần (plain text). Không dùng cú pháp Markdown như **, *, #, backtick hoặc bảng Markdown.

Chế độ trả lời:
{instruction}

QUY TẮC VỀ NGUỒN:
- Luôn kết thúc bằng mục "Nguồn" thật ngắn, tối đa 2-3 dòng và chỉ liệt kê nguồn thực sự dùng.
- Không chèn nhãn "(Nguồn N)" hoặc trích dẫn nguồn xen giữa phần trả lời; chỉ ghi nguồn ở mục cuối.
- Với giáo trình, chỉ ghi tên rút gọn, chương và trang; ví dụ: "- Giáo trình, Chương 2, tr.53-68".
- Với văn bản pháp luật, chỉ ghi tên/số hiệu văn bản và điều; không chép URL hay tiêu đề đoạn dài.
- Không ghi lại tên đề mục, subsection hoặc nội dung trích đoạn trong danh sách nguồn.
- Mỗi đoạn ngữ cảnh bên dưới có nhãn "[Nguồn N] ...". Chỉ được trích đúng thông tin nguồn ghi trong nhãn đó.
- TUYỆT ĐỐI không tự ghi số trang giáo trình nếu trang đó không xuất hiện ở nhãn [Nguồn N] nào.
- Nếu thông tin lấy từ "Bản tổng hợp RAG" (nhãn không có số trang), hãy trích đúng "(Bản tổng hợp RAG)", KHÔNG suy ra số trang giáo trình từ nội dung bản tổng hợp.
- Ví dụ định dạng hợp lệ: (Giáo trình, Ch.4, tr.116) — chỉ khi có nhãn tương ứng; (Luật Điện lực 61/2024, Điều 5); (VBHN 51/VBHN-BXD năm 2026, Điều 3); (Bản tổng hợp RAG).

Guardrails:
{guardrails}
"""


def build_messages(
    question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    mode: str = "default",
    pinned: str | None = None,
) -> list[dict[str, str]]:
    pinned_block = ""
    if pinned and pinned.strip():
        pinned_block = (
            "Đoạn người dùng đang bôi đen trên slide (context ưu tiên — hãy bám sát và giải thích chính đoạn này, "
            "rồi dùng ngữ cảnh truy hồi bên dưới để bổ sung/đối chiếu/trích nguồn):\n"
            f"\"\"\"\n{pinned.strip()}\n\"\"\"\n\n"
        )

    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["default"])
    user_prompt = f"""{pinned_block}Câu hỏi:
{question}

Ngữ cảnh truy hồi:
{context_block(contexts)}

Hãy trả lời câu hỏi dựa trên ngữ cảnh trên.
Cuối câu trả lời, thêm mục "Nguồn" gồm tối đa 2-3 dòng ngắn. Chỉ ghi tên nguồn rút gọn, chương/trang hoặc điều luật; không sao chép tiêu đề đề mục dài từ nhãn [Nguồn N].

YÊU CẦU VỀ CÁCH TRẢ LỜI (BẮT BUỘC theo đúng chế độ đã chọn, đặt lên trên mọi thói quen định dạng khác):
{instruction}

Chỉ trả về văn bản thuần; tuyệt đối không dùng dấu ** để in đậm."""

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt_for_mode(mode)}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_llm(
    messages: list[dict[str, str]], stream: bool = False, temperature: float = 0.2
) -> str | Iterator[str]:
    client = config.get_openai_client()

    if stream:
        response = config.with_backoff(
            lambda: client.chat.completions.create(
                model=config.GENERATION_MODEL,
                temperature=temperature,
                messages=messages,
                stream=True,
            )
        )

        def deltas() -> Iterator[str]:
            pending_star = False
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    output: list[str] = []
                    for char in delta:
                        if char == "*":
                            if pending_star:
                                pending_star = False
                            else:
                                pending_star = True
                            continue
                        if pending_star:
                            output.append("*")
                            pending_star = False
                        output.append(char)
                    if output:
                        yield "".join(output)
            if pending_star:
                yield "*"

        return deltas()

    response = config.with_backoff(
        lambda: client.chat.completions.create(
            model=config.GENERATION_MODEL,
            temperature=temperature,
            messages=messages,
        )
    )
    return (response.choices[0].message.content or "").replace("**", "")


def answer(
    question: str,
    history: list[dict[str, str]] | None = None,
    mode: str = "default",
) -> tuple[str, list[dict[str, Any]]]:
    contexts = hybrid_retrieve(question)
    messages = build_messages(question, contexts, history=history, mode=mode)
    text = call_llm(messages, stream=False, temperature=temperature_for_mode(mode))
    return str(text), contexts


def verify_citations(text: str, contexts: list[dict[str, Any]]) -> list[str]:
    """Flag textbook page citations that are not backed by retrieved context.

    Catches the case where the model invents a page like "(Giáo trình, Ch.4,
    tr.999)" that was never in the retrieved passages. Returns warning strings.
    """

    covered: set[int] = set()
    for context in contexts:
        if context.get("source_type") != "textbook":
            continue
        start, end = context.get("page_start"), context.get("page_end")
        if isinstance(start, int) and isinstance(end, int):
            covered.update(range(start, end + 1))

    warnings: list[str] = []
    for match in re.finditer(r"tr\.\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?", text):
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        cited = set(range(min(first, last), max(first, last) + 1))
        if not cited & covered:
            warnings.append(
                f"Trích dẫn tr.{match.group(0).split('.', 1)[1].strip()} không nằm trong ngữ cảnh đã truy hồi "
                f"(các trang có trong ngữ cảnh: {sorted(covered) or 'không có trang giáo trình'})."
            )
    return warnings


def print_answer(question: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    text, contexts = answer(question, history=history)
    print("\n" + text.strip())

    warnings = verify_citations(text, contexts)
    if warnings:
        print("\n⚠️  Cảnh báo trích dẫn:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\n--- Retrieved contexts ---")
    for idx, context in enumerate(contexts, start=1):
        print(f"[{idx}] {format_source(context)}")

    new_history = list(history or [])
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": text})
    return new_history[-6:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Question to ask. If omitted, starts an interactive loop.")
    args = parser.parse_args()

    history: list[dict[str, str]] = []
    if args.question:
        print_answer(" ".join(args.question), history=None)
        return

    print("Chatbot RAG chủ đề độc quyền. Gõ 'exit' để thoát.")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        history = print_answer(question, history=history)


if __name__ == "__main__":
    main()
