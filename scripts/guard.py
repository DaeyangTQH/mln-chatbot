"""Chốt chặn đầu vào (input triage) trước khi chạy RAG.

Bắt các trường hợp KHÔNG nên đưa vào retrieval + generation:
  - injection / jailbreak (kể cả giấu trong pinned text),
  - câu rỗng / chỉ emoji / gõ loạn phím (gibberish),
  - đại từ mồ côi ("nó", "cái đó"...) không có tham chiếu.

Phần lớn là luật xác định (regex/heuristic) nên rẻ và ổn định — cùng một đầu vào
luôn cho cùng một hành vi, thay vì lúc trả lời lúc chế bài như trước.
"""

from __future__ import annotations

import re
from typing import Any

# Câu trả lời cố định cho từng loại chặn — KHÔNG kèm mục "Nguồn" (không có nguồn để trích).

# Ngoài KHO TÀI LIỆU hoàn toàn (thời tiết, lập trình, lịch sử...) — cổng relevance thấp.
OUT_OF_CORPUS_TEXT = (
    "Nội dung này không có trong kho tài liệu tôi được cung cấp (chỉ gồm chủ đề độc quyền "
    "trong Kinh tế chính trị Mác - Lênin và các văn bản điện, nước ở Việt Nam). "
    "Bạn thử hỏi một câu thuộc phạm vi đó nhé."
)

# Thuộc môn/kinh tế nhưng NGOÀI CHỦ ĐỀ độc quyền (giá trị thặng dư, địa tô...) — classifier loại.
OUT_OF_TOPIC_TEXT = (
    "Câu hỏi này nằm ngoài phạm vi chủ đề tôi phụ trách. Kho tài liệu hiện chỉ bao gồm phần "
    "độc quyền trong Kinh tế chính trị Mác - Lênin (cạnh tranh và độc quyền, độc quyền nhà "
    "nước, tích tụ/tập trung tư bản, xuất khẩu tư bản...) cùng một số văn bản điện, nước. "
    "Bạn hỏi trong phạm vi này nhé."
)

CLARIFY_TEXT = (
    "Tôi chưa rõ bạn muốn hỏi gì. Bạn nêu cụ thể câu hỏi về chủ đề độc quyền "
    "(khái niệm, nguyên nhân, đặc điểm, độc quyền nhà nước, hay vận dụng ở Việt Nam...) "
    "để tôi hỗ trợ chính xác nhé."
)

INJECTION_TEXT = (
    "Tôi không thể làm theo yêu cầu thay đổi vai trò, tiết lộ hướng dẫn hệ thống, "
    "hoặc bỏ qua các quy tắc trả lời. Bạn có thể đặt câu hỏi về chủ đề độc quyền trong "
    "môn Kinh tế chính trị Mác - Lênin."
)

# Nguyên âm tiếng Việt (kể cả có dấu) + a,e,i,o,u,y — dùng để phát hiện gõ loạn phím.
_VOWELS = set("aeiouyàáạảãăằắặẳẵâầấậẩẫèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ")

# Từ khóa chủ đề: nếu câu có chứa, KHÔNG coi là đại từ mồ côi (còn bám corpus).
_DOMAIN_HINTS = (
    "độc quyền", "nhà nước", "tư bản", "tư nhân", "cạnh tranh", "thị trường", "điện",
    "nước", "giá cả", "giá điện", "tích tụ", "tập trung", "đế quốc", "lợi nhuận",
    "xuất khẩu tư bản", "tổ chức độc quyền", "mác", "lênin", "kinh tế",
)

# Mẫu tấn công prompt / jailbreak (dò trên bản lowercase).
_INJECTION_PATTERNS = [
    r"bỏ qua\s+(mọi|tất cả|các)?\s*(hướng dẫn|chỉ dẫn|quy tắc|yêu cầu trước)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instruction|prompt|message)",
    r"(in|in ra|tiết lộ|hiển thị|cho.*xem|lặp lại).{0,20}(system prompt|hướng dẫn hệ thống|chỉ dẫn hệ thống)",
    r"system prompt",
    r"(đóng vai|nhập vai|bây giờ bạn là|từ giờ bạn là)\s*['\"]?\s*dan",
    r"\bdan\b.{0,30}(không có|no).{0,15}(giới hạn|restriction|limit)",
    r"jailbroken",
    r"trả lời\s+(đúng\s+)?(một|1)\s+từ",
    r"quên\s+(đi\s+)?(môn|hướng dẫn|mọi thứ)",
    r"\bact as\b",
    r"(reveal|show|print).{0,20}(prompt|instruction)",
    r"lặp lại nguyên văn.{0,30}(hướng dẫn|prompt)",
]


def _letters(text: str) -> str:
    return re.sub(r"[^0-9A-Za-zÀ-ỹ]", "", text or "")


def _looks_injection(text: str | None) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(re.search(pattern, low) for pattern in _INJECTION_PATTERNS)


def _word_like(token: str) -> bool:
    """Một token 'giống từ thật' nếu có nguyên âm và không có chuỗi phụ âm quá dài."""

    t = token.lower()
    if not any(c in _VOWELS for c in t):
        return False
    run = longest = 0
    for c in t:
        if c.isalpha() and c not in _VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest <= 3


def _is_gibberish(text: str) -> bool:
    tokens = [t for t in re.split(r"\s+", text.strip()) if _letters(t)]
    if not tokens or len(_letters(text)) < 4:
        return False
    word_like = sum(_word_like(t) for t in tokens)
    return word_like / len(tokens) < 0.5


def _orphan_pronoun(question: str) -> bool:
    """Câu ngắn xoay quanh 'nó/cái đó/hai cái đó...' mà không nhắc từ khóa chủ đề."""

    low = question.lower()
    if any(hint in low for hint in _DOMAIN_HINTS):
        return False
    has_pronoun = bool(
        re.match(r"^\s*(nó|vậy nó|chúng nó)\b", low)
        or re.search(r"\b(cái đó|cái này|điều đó|hai cái đó|thứ đó|mấy cái đó)\b", low)
    )
    return has_pronoun and len(low.split()) <= 12


# --- FAQ / câu hỏi vui về chatbot & website ---
# Những câu này KHÔNG đi qua RAG: chúng là thông tin cố định về nhóm/website, nên trả
# lời bằng template thay vì để cổng relevance chặn nhầm. Regex cố ý HẸP (bắt buộc có từ
# khóa web/website/nhóm/chatbot/bạn) để không nuốt nhầm câu học thật kiểu "mục tiêu của
# độc quyền nhà nước" hay "chủ đề độc quyền".
#
# TODO(nhóm): thay các chỗ «...» bằng thông tin thật rồi bỏ dấu ngoặc.
_META_INTRO = (
    "Mình là trợ giảng ảo của website học phần Kinh tế chính trị Mác - Lênin (MLN121), "
    "chuyên hỗ trợ ôn tập chủ đề độc quyền và độc quyền nhà nước. Bạn có thể hỏi mình về "
    "khái niệm, nguyên nhân, đặc điểm của độc quyền, hoặc phần vận dụng ở Việt Nam nhé."
)

_META_GROUP = (
    "Mình được xây dựng bởi nhóm 6, gồm các thành viên cực đẳng cấp. "
    "Đây là dự án học phần Kinh tế chính trị Mác - Lênin (MLN121)."
)

_META_TOPIC = (
    "Chủ đề của nhóm là ĐỘC QUYỀN trong Kinh tế chính trị Mác - Lênin: cạnh tranh và độc "
    "quyền, độc quyền nhà nước, nguyên nhân và đặc điểm của độc quyền, cùng phần vận dụng "
    "vào ngành điện, nước ở Việt Nam."
)

_META_GOAL = (
    "Website được lập ra để giúp sinh viên học và ôn tập chủ đề độc quyền trong môn Kinh tế "
    "chính trị Mác - Lênin một cách trực quan: đọc nội dung theo từng chương, tra cứu nhanh "
    "bằng chatbot, và tự kiểm tra qua quiz/flashcard. «bổ sung mục tiêu cụ thể của nhóm nếu có»."
)

_META_SOURCES = (
    "Nội dung website dựa trên Giáo trình Kinh tế chính trị Mác - Lênin (Bộ Giáo dục và Đào "
    "tạo, 2021), kết hợp một số văn bản pháp luật hiện hành về điện và nước ở Việt Nam (ví dụ "
    "Luật Điện lực 2024, văn bản về cấp nước sạch). Chatbot chỉ trả lời dựa trên các nguồn này."
)

_META_SECTIONS = (
    "Website gồm phần giới thiệu và các chương nội dung về độc quyền: khái niệm và nguyên nhân "
    "hình thành độc quyền, độc quyền nhà nước, các lĩnh vực/ngành liên quan, phần tranh luận - "
    "phản biện, và phần vận dụng ở Việt Nam. Ngoài ra có công cụ tóm tắt, quiz, flashcard và "
    "chatbot hỏi đáp."
)

# Mỗi mục: (regex, câu trả lời). Xét theo thứ tự, khớp cái đầu tiên là dừng.
_META_PATTERNS: list[tuple[str, str]] = [
    (r"\b(bạn|em|chatbot|bot|trợ giảng)\b.{0,20}\b(là ai|là gì|tên (gì|là gì))"
     r"|\bgiới thiệu\b.{0,15}\b(bản thân|về (bạn|mình|chatbot|bot))\b"
     r"|\bbạn\b.{0,10}\btự giới thiệu\b", _META_INTRO),
    (r"\bnhóm\b.{0,15}\b(nào|mấy|số mấy|tên (gì|là gì)|là ai|gồm (những )?ai|bao nhiêu (người|thành viên))"
     r"|\b(thành viên|ai làm|ai xây dựng|ai tạo)\b.{0,15}\bnhóm\b", _META_GROUP),
    (r"\bchủ đề\b.{0,15}\b(của )?(nhóm|website|web|trang|dự án|các bạn|nhóm bạn)"
     r"|\bđề tài\b.{0,15}\b(của )?(nhóm|các bạn|nhóm bạn|website|web)", _META_TOPIC),
    (r"\bmục (tiêu|đích)\b.{0,15}\b(của )?(website|web|trang|dự án|nhóm|các bạn)"
     r"|\b(website|web|trang|dự án)\b.{0,10}\b(để làm gì|nhằm|lập ra|tạo ra để)", _META_GOAL),
    (r"\b(nguồn|tài liệu|dữ liệu)\b.{0,25}\b(website|web|trang|lấy từ đâu|từ đâu|ở đâu|dựa (trên|vào))"
     r"|\btài liệu\b.{0,10}\blấy\b", _META_SOURCES),
    (r"\b(website|web|trang)\b.{0,25}\b(phần nào|những phần|gồm (những )?(gì|phần)|mục nào|có (những )?gì|nội dung gì)"
     r"|\bcó những phần nào\b|\bgồm những mục\b", _META_SECTIONS),
]


def meta_response(question: str | None) -> dict[str, str] | None:
    """Trả về câu FAQ cố định về chatbot/website nếu câu hỏi thuộc loại meta, ngược lại None."""

    if not question:
        return None
    low = question.lower()
    for pattern, text in _META_PATTERNS:
        if re.search(pattern, low):
            return {"kind": "meta", "text": text}
    return None


def triage(
    question: str,
    history: list[dict[str, Any]] | None = None,
    pinned: str | None = None,
) -> dict[str, str] | None:
    """Trả về {"kind","text"} nếu nên chặn, hoặc None nếu để pipeline xử lý bình thường."""

    q = (question or "").strip()

    # 1) Injection có mức ưu tiên cao nhất — dò cả trong câu hỏi lẫn pinned.
    if _looks_injection(q) or _looks_injection(pinned):
        return {"kind": "injection", "text": INJECTION_TEXT}

    # 2) Rỗng / chỉ dấu câu-emoji / gõ loạn / spam ký tự.
    letters = _letters(q)
    if len(letters) < 2:
        return {"kind": "clarify", "text": CLARIFY_TEXT}
    if re.fullmatch(r"(.)\1{4,}", q.replace(" ", "")):
        return {"kind": "clarify", "text": CLARIFY_TEXT}
    if _is_gibberish(q):
        return {"kind": "clarify", "text": CLARIFY_TEXT}

    # 3) Đại từ mồ côi — chỉ chặn khi KHÔNG có lịch sử/pinned để giải nghĩa.
    if not history and not (pinned and pinned.strip()) and _orphan_pronoun(q):
        return {"kind": "clarify", "text": CLARIFY_TEXT}

    return None
