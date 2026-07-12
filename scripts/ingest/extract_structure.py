"""Extract a structure-aware corpus from the textbook PDF and curated sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config


@dataclass(frozen=True)
class PageRange:
    start: int
    end: int
    chapter: str
    title: str
    relevance: str


PAGE_RANGES = [
    PageRange(
        14,
        14,
        "Chương 1",
        "Đối tượng, phương pháp nghiên cứu và chức năng của kinh tế chính trị Mác - Lênin",
        "Bối cảnh đóng góp của V.I. Lênin về độc quyền và độc quyền nhà nước.",
    ),
    PageRange(
        49,
        73,
        "Chương 2",
        "Hàng hóa, thị trường và vai trò của các chủ thể tham gia thị trường",
        "Nền tảng về thị trường, cạnh tranh, khuyết tật thị trường và vai trò Nhà nước.",
    ),
    PageRange(
        112,
        151,
        "Chương 4",
        "Cạnh tranh và độc quyền trong nền kinh tế thị trường",
        "Nguồn lõi về độc quyền, nguyên nhân hình thành độc quyền và độc quyền nhà nước.",
    ),
    PageRange(
        152,
        176,
        "Chương 5",
        "Kinh tế thị trường định hướng xã hội chủ nghĩa và các quan hệ lợi ích kinh tế ở Việt Nam",
        "Vận dụng ở Việt Nam: Nhà nước, kinh tế nhà nước, kinh tế tư nhân, lĩnh vực then chốt.",
    ),
]

OPTIONAL_CH5_PART3 = PageRange(
    177,
    200,
    "Chương 5",
    "Kinh tế thị trường định hướng xã hội chủ nghĩa và các quan hệ lợi ích kinh tế ở Việt Nam",
    "Phần III về quan hệ lợi ích kinh tế, chỉ bật khi cần mở rộng corpus.",
)

SECTION_RE = re.compile(r"^([IVX]+)-\s*(.+)?$")
SUBSECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")
CHAPTER_RE = re.compile(r"^Chương\s+\d+\s*$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
FOOTNOTE_SEPARATOR_RE = re.compile(r"^_+$")
TAIL_HEADING_RE = re.compile(r"^(TÓM TẮT CHƯƠNG|CÁC THUẬT NGỮ|VẤN ĐỀ THẢO LUẬN|CÂU HỎI ÔN TẬP)", re.IGNORECASE)


POLICY_NODES = [
    {
        "source_type": "policy",
        "source_title": "Luật Điện lực số 61/2024/QH15",
        "source_url": "https://xaydungchinhsach.chinhphu.vn/toan-van-luat-dien-luc-119241225130808137.htm",
        "citation": "Luật Điện lực 61/2024, Điều 5, Điều 50",
        "chapter": None,
        "section_path": "Nguồn chính sách > Ngành điện",
        "breadcrumb": "Nguồn chính sách > Luật Điện lực 61/2024 > Điều 5, Điều 50",
        "page_start": None,
        "page_end": None,
        "text": (
            "Luật Điện lực số 61/2024/QH15 xác định Nhà nước độc quyền một số hoạt động "
            "vì an ninh năng lượng quốc gia, gồm điều độ hệ thống điện quốc gia, đầu tư xây dựng "
            "và vận hành một số dự án điện hạt nhân, thủy điện chiến lược đa mục tiêu, lưới truyền tải "
            "quan trọng từ cấp điện áp 220 kV trở lên theo danh mục do Thủ tướng quyết định, và vận hành "
            "lưới truyền tải trừ phần do thành phần kinh tế ngoài nhà nước đầu tư xây dựng. Luật cũng thu hút "
            "mọi thành phần kinh tế tham gia đầu tư nguồn điện, lưới điện, phát điện, phân phối điện, bán buôn "
            "và bán lẻ điện theo quy hoạch; xây dựng thị trường điện cạnh tranh có sự điều tiết của Nhà nước; "
            "giá bán điện theo cơ chế thị trường có sự điều tiết giá của Nhà nước."
        ),
    },
    {
        "source_type": "policy",
        "source_title": "Văn bản hợp nhất số 51/VBHN-BXD năm 2026 về sản xuất, cung cấp và tiêu thụ nước sạch",
        "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/6/51-vbhn-bxd.pdf",
        "citation": "VBHN 51/VBHN-BXD năm 2026, Điều 3, Điều 30-32, Điều 51",
        "chapter": None,
        "section_path": "Nguồn chính sách > Ngành nước",
        "breadcrumb": "Nguồn chính sách > VBHN 51/VBHN-BXD năm 2026 > Điều 3, Điều 30-32, Điều 51",
        "page_start": None,
        "page_end": None,
        "text": (
            "Văn bản hợp nhất số 51/VBHN-BXD năm 2026 nêu hoạt động cấp nước là loại hình hoạt động "
            "sản xuất kinh doanh chịu sự kiểm soát của Nhà nước nhằm bảo đảm quyền và lợi ích hợp pháp "
            "của đơn vị cấp nước và khách hàng sử dụng nước, có xét đến hỗ trợ cấp nước cho người nghèo "
            "và khu vực đặc biệt khó khăn. Văn bản khuyến khích các thành phần kinh tế, cộng đồng xã hội "
            "tham gia đầu tư phát triển và quản lý hoạt động cấp nước. Mỗi vùng phục vụ cấp nước chỉ do một "
            "đơn vị cấp nước thực hiện dịch vụ; giá nước sạch thực hiện theo Luật Giá và nằm trong khung giá, "
            "biểu giá nước do Nhà nước quy định, có xét tới duy trì dịch vụ, phát triển, tiết kiệm nước và hỗ trợ "
            "người nghèo."
        ),
    },
]


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = line.replace("\x00", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def should_skip_line(line: str, page_num: int) -> bool:
    if not line:
        return True
    if line == str(page_num) or PAGE_NUMBER_RE.match(line):
        return True
    if FOOTNOTE_SEPARATOR_RE.match(line):
        return True
    if CHAPTER_RE.match(line):
        return True
    return False


def repair_leading_fragment(lines: list[str], page_num: int) -> list[str]:
    if page_num == 14 and lines and lines[0].startswith("bật là kết quả"):
        lines[0] = "Nổi bật là kết quả" + lines[0][len("bật là kết quả") :]
    return lines


def extract_page_lines(pdf: pdfplumber.PDF, page_num: int) -> list[str]:
    text = pdf.pages[page_num - 1].extract_text(x_tolerance=1, y_tolerance=3) or ""
    lines = []
    in_footnote = False
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if FOOTNOTE_SEPARATOR_RE.match(line):
            in_footnote = True
            continue
        if in_footnote:
            continue
        if not should_skip_line(line, page_num):
            lines.append(line)
    return repair_leading_fragment(lines, page_num)


def uppercase_ratio(line: str) -> float:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def is_probable_chapter_title(line: str) -> bool:
    return uppercase_ratio(line) > 0.72 and len(line) <= 90 and not SECTION_RE.match(line)


def is_heading_continuation(line: str, kind: str, current_title: str) -> bool:
    if SECTION_RE.match(line) or SUBSECTION_RE.match(line) or CHAPTER_RE.match(line):
        return False
    if kind == "section":
        return uppercase_ratio(line) > 0.55 and not line.endswith((".", ":", ";"))
    if kind == "subsection":
        if line and line[0].islower() and not line.endswith((".", ":", ";")):
            return True
        return len(current_title) < 90 and uppercase_ratio(line) > 0.55
    return False


def paragraph_text(lines: list[str]) -> str:
    if not lines:
        return ""

    paragraphs: list[str] = []
    current = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not current:
            current = line
            continue

        starts_new = bool(re.match(r"^([-+*]|\d+\)|[a-zđ]\)|[A-ZĐ]\))\s+", line))
        previous_open = not current.endswith((".", "!", "?", ":", ";", "…”", '."', ')"'))
        current_lower = bool(line and line[0].islower())

        if current.endswith("-") and current_lower:
            current = current[:-1] + line
        elif starts_new and not current_lower:
            paragraphs.append(current)
            current = line
        elif current_lower or previous_open:
            current += " " + line
        else:
            paragraphs.append(current)
            current = line

    if current:
        paragraphs.append(current)

    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_breadcrumb(chapter: str, chapter_title: str, section: str | None, subsection: str | None) -> str:
    parts = [f"{chapter}: {chapter_title}"]
    if section:
        parts.append(section)
    if subsection:
        parts.append(subsection)
    return " > ".join(parts)


def extract_textbook_nodes(include_ch5_part3: bool = False) -> list[dict[str, Any]]:
    ranges = PAGE_RANGES + ([OPTIONAL_CH5_PART3] if include_ch5_part3 else [])
    nodes: list[dict[str, Any]] = []
    # Pages where the chapter heading actually appears -> a real chapter intro.
    title_pages: set[int] = set()

    with pdfplumber.open(str(config.PDF_PATH)) as pdf:
        for page_range in ranges:
            current_section: str | None = None
            current_subsection: str | None = None
            current_node: dict[str, Any] | None = None
            pending_heading: tuple[str, dict[str, Any]] | None = None

            def flush_node() -> None:
                nonlocal current_node
                if not current_node:
                    return
                text = paragraph_text(current_node.pop("_lines"))
                if text:
                    current_node["text"] = text
                    current_node["breadcrumb"] = make_breadcrumb(
                        current_node["chapter"],
                        current_node["chapter_title"],
                        current_node.get("section"),
                        current_node.get("subsection"),
                    )
                    current_node["section_path"] = " > ".join(
                        part
                        for part in [current_node.get("section"), current_node.get("subsection")]
                        if part
                    ) or "Giới thiệu chương"
                    nodes.append(current_node)
                current_node = None

            def start_node(page_num: int, section: str | None, subsection: str | None) -> dict[str, Any]:
                nonlocal current_node
                flush_node()
                current_node = {
                    "source_type": "textbook",
                    "source_title": "Giáo trình Kinh tế chính trị Mác - Lênin, Bộ Giáo dục và Đào tạo, 2021",
                    "source_file": config.PDF_PATH.name,
                    "source_url": None,
                    "citation": None,
                    "chapter": page_range.chapter,
                    "chapter_title": page_range.title,
                    "section": section,
                    "subsection": subsection,
                    "page_start": page_num,
                    "page_end": page_num,
                    "_lines": [],
                }
                return current_node

            current_node = start_node(page_range.start, None, None)
            skip_tail = False

            for page_num in range(page_range.start, page_range.end + 1):
                for line in extract_page_lines(pdf, page_num):
                    if TAIL_HEADING_RE.match(line):
                        skip_tail = True
                        continue
                    if skip_tail:
                        continue

                    if is_probable_chapter_title(line) and page_num == page_range.start:
                        title_pages.add(page_num)
                        continue

                    if pending_heading:
                        kind, node = pending_heading
                        title_key = "section" if kind == "section" else "subsection"
                        title = node.get(title_key) or ""
                        if is_heading_continuation(line, kind, title):
                            node[title_key] = f"{title} {line}".strip()
                            if kind == "section":
                                current_section = node[title_key]
                            else:
                                current_subsection = node[title_key]
                            node["page_end"] = page_num
                            continue
                        pending_heading = None

                    section_match = SECTION_RE.match(line)
                    if section_match:
                        current_section = f"{section_match.group(1)}- {(section_match.group(2) or '').strip()}".strip()
                        current_subsection = None
                        node = start_node(page_num, current_section, None)
                        pending_heading = ("section", node)
                        continue

                    subsection_match = SUBSECTION_RE.match(line)
                    if subsection_match:
                        current_subsection = f"{subsection_match.group(1)}. {subsection_match.group(2).strip()}"
                        node = start_node(page_num, current_section, current_subsection)
                        pending_heading = ("subsection", node)
                        continue

                    if current_node is None:
                        current_node = start_node(page_num, current_section, current_subsection)
                    current_node["_lines"].append(line)
                    current_node["page_end"] = page_num

            flush_node()

    nodes = [node for node in nodes if keep_intro_node(node, title_pages)]
    for idx, node in enumerate(nodes, start=1):
        node["id"] = f"textbook_node_{idx:04d}"
    return nodes


# Keywords that mark an intro paragraph as genuinely on-topic (not page spillover).
INTRO_CORE_KEYWORDS = ("độc quyền", "cạnh tranh", "khuyết tật", "nhà nước")


def keep_intro_node(node: dict[str, Any], title_pages: set[int]) -> bool:
    """Drop "Giới thiệu chương" nodes that are spillover from the previous chapter.

    A chapter-intro node is kept only if the chapter heading appears on its first
    page (a real intro) OR its text contains a core monopoly/market keyword. This
    removes the page-48 tail (chứng khoán/tiền) that leaks onto page 49, while
    keeping the deliberate Lênin context on page 14.
    """

    if node.get("section_path") != "Giới thiệu chương":
        return True
    if node.get("page_start") in title_pages:
        return True
    body = (node.get("text") or "").lower()
    return any(keyword in body for keyword in INTRO_CORE_KEYWORDS)


def extract_synthesis_nodes() -> list[dict[str, Any]]:
    if not config.SYNTHESIS_PATH.exists():
        return []

    text = config.SYNTHESIS_PATH.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))
    nodes: list[dict[str, Any]] = []
    for idx, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[idx].start() if idx < len(matches) else len(text)
        section = match.group(1).strip()
        body = text[start:end].strip()
        nodes.append(
            {
                "id": f"synthesis_node_{idx:04d}",
                "source_type": "synthesis",
                "source_title": "Bản tổng hợp RAG: độc quyền, độc quyền nhà nước và vận dụng điện - nước ở Việt Nam",
                "source_file": config.SYNTHESIS_PATH.name,
                "source_url": None,
                "citation": "Bản tổng hợp RAG",
                "chapter": None,
                "chapter_title": None,
                "section": section,
                "subsection": None,
                "section_path": section,
                "breadcrumb": f"Bản tổng hợp > {section}",
                "page_start": None,
                "page_end": None,
                "text": body,
            }
        )
    return nodes


def extract_policy_nodes() -> list[dict[str, Any]]:
    nodes = []
    for idx, raw in enumerate(POLICY_NODES, start=1):
        node = dict(raw)
        node["id"] = f"policy_node_{idx:04d}"
        node["chapter_title"] = None
        node["section"] = node["section_path"]
        node["subsection"] = None
        node["source_file"] = None
        nodes.append(node)
    return nodes


def build_structured(include_ch5_part3: bool = False) -> dict[str, Any]:
    textbook_nodes = extract_textbook_nodes(include_ch5_part3=include_ch5_part3)
    synthesis_nodes = extract_synthesis_nodes()
    policy_nodes = extract_policy_nodes()
    return {
        "source_pdf": str(config.PDF_PATH),
        "page_ranges": [page_range.__dict__ for page_range in PAGE_RANGES],
        "include_ch5_part3": include_ch5_part3,
        "nodes": textbook_nodes + synthesis_nodes + policy_nodes,
        "stats": {
            "textbook_nodes": len(textbook_nodes),
            "synthesis_nodes": len(synthesis_nodes),
            "policy_nodes": len(policy_nodes),
            "total_nodes": len(textbook_nodes) + len(synthesis_nodes) + len(policy_nodes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ch5-part3", action="store_true", default=config.INCLUDE_CHAPTER5_PART3)
    args = parser.parse_args()

    config.ensure_output_dir()
    structured = build_structured(include_ch5_part3=args.include_ch5_part3)
    config.STRUCTURED_PATH.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(config.STRUCTURED_PATH), **structured["stats"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
