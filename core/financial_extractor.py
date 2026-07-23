"""financial_extractor.py — Extract financial statement line items from annual report text."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Vietnamese financial statement line-item patterns
# Each pattern maps: display_name -> (regex_pattern, multiplier)
# The regex matches the line item label; the number is the first numeric value
# following the label on the same line or next non-blank line.

# Line-item patterns as plain lowercase keywords (matched with re.search on lowercased lines).
# Order matters: more specific patterns first to avoid false matches.

# Vietnamese financial keywords for line-item matching (actual Vietnamese, with diacritics).
# These are normalized to a searchable form before matching.

_INCOME_STMT_ITEMS: list[tuple[str, str]] = [
    ("doanh_thu_thuan", "doanh thu bán hàng và cung cấp dịch vụ"),
    ("doanh_thu_thuan", "doanh thu thuần về"),
    ("doanh_thu_thuan", "doanh thu bán hàng"),
    ("doanh_thu_thuan", "doanh thu thuần"),
    ("doanh_thu_thuan", "doanh thu"),
    ("gia_von", "giá vốn hàng bán"),
    ("loi_nhuan_gop", "lợi nhuận gộp"),
    ("doanh_thu_tai_chinh", "doanh thu tài chính"),
    ("chi_phi_tai_chinh", "chi phí tài chính"),
    ("chi_phi_ban_hang", "chi phí bán hàng"),
    ("chi_phi_ql", "chi phí quản lý doanh nghiệp"),
    ("loi_nhuan_hdkd", "lợi nhuận thuần từ hoạt động kinh doanh"),
    ("loi_nhuan_hdkd", "lợi nhuận từ hoạt động kinh doanh"),
    ("loi_nhuan_truoc_thue", "lợi nhuận trước thuế"),
    ("loi_nhuan_sau_thue", "lợi nhuận sau thuế thu nhập doanh nghiệp"),
    ("loi_nhuan_sau_thue", "lợi nhuận sau thuế của công ty mẹ"),
    ("loi_nhuan_sau_thue", "lợi nhuận sau thuế"),
    ("loi_nhuan_sau_thue", "lợi nhuận ròng"),
    ("loi_nhuan_co_dong", "lợi nhuận của công ty mẹ"),
    ("ebitda", "ebitda"),
]

_BALANCE_SHEET_ITEMS: list[tuple[str, str]] = [
    ("tai_san_ngan_han", "tài sản ngắn hạn"),
    ("tien_va_tuong_duong", "tiền và các khoản tương đương tiền"),
    ("tien_va_tuong_duong", "tiền"),
    ("phai_thu", "các khoản phải thu"),
    ("hang_ton_kho", "hàng tồn kho"),
    ("tai_san_dai_han", "tài sản dài hạn"),
    ("tai_san_co_dinh", "tài sản cố định"),
    ("tong_tai_san", "tổng cộng tài sản"),
    ("tong_tai_san", "tổng tài sản"),
    ("no_ngan_han", "nợ ngắn hạn"),
    ("no_dai_han", "nợ dài hạn"),
    ("no_vay", "vay và nợ thuê tài chính"),
    ("no_vay_ngan_han", "vay ngắn hạn"),
    ("no_vay_dai_han", "vay dài hạn"),
    ("von_chu_so_huu", "vốn chủ sở hữu"),
    ("tong_no", "tổng cộng nợ"),
    ("tong_cong_no", "tổng cộng nợ phải trả"),
]

_CASH_FLOW_ITEMS: list[tuple[str, str]] = [
    ("luu_chuyen_tu_hdkd", "lưu chuyển tiền thuần từ hoạt động kinh doanh"),
    ("luu_chuyen_tu_hdkd", "lưu chuyển tiền từ hoạt động kinh doanh"),
    ("khau_hao", "khấu hao"),
    ("lai_vay", "lãi vay"),
    ("thay_doi_von_luudong", "thay đổi vốn lưu động"),
    ("luu_chuyen_tu_hd dt", "lưu chuyển tiền từ hoạt động đầu tư"),
    ("chi_mua_tscd", "mua sắm tài sản cố định"),
    ("chi_mua_tscd", "chi phí xây dựng cơ bản"),
    ("luu_chuyen_tu_hdtc", "lưu chuyển tiền từ hoạt động tài chính"),
    ("luu_chuyen_thuan", "lưu chuyển tiền thuần trong kỳ"),
    ("tien_cuoi_ky", "tiền và tương đương tiền cuối kỳ"),
]

# Known multiplier suffixes
def _parse_number(text: str) -> float | None:
    """Extract the first number from a line of text."""
    text = text.replace(",", "").replace(" ", "").replace("–", "-")
    match = re.search(r"[-]?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def extract_financials(report_text: str) -> dict[str, Any]:
    """Extract key financial line items from annual report text.

    Returns a dict with keys:
        - income_statement: list of {label, values}
        - balance_sheet: list of {label, values}
        - cash_flow: list of {label, values}
        - sections_detected: dict of section_name -> bool
    """
    lines = report_text.split("\n")
    sections = _detect_sections(report_text)

    return {
        "income_statement": _extract_section(lines, _INCOME_STMT_ITEMS),
        "balance_sheet": _extract_section(lines, _BALANCE_SHEET_ITEMS),
        "cash_flow": _extract_section(lines, _CASH_FLOW_ITEMS),
        "sections_detected": sections,
    }


_SECTION_MARKERS: list[tuple[str, str]] = [
    ("income_statement", "kết quả hoạt động kinh doanh"),
    ("income_statement", "báo cáo kết quả"),
    ("income_statement", "income statement"),
    ("balance_sheet", "bảng cân đối kế toán"),
    ("balance_sheet", "balance sheet"),
    ("balance_sheet", "tình hình tài chính"),
    ("cash_flow", "lưu chuyển tiền tệ"),
    ("cash_flow", "báo cáo lưu chuyển tiền tệ"),
    ("cash_flow", "cash flow"),
]


def _detect_sections(text: str) -> dict[str, bool]:
    """Detect which financial statement sections exist in the text."""
    stripped = _strip_diacritics(text).lower()
    found: dict[str, bool] = {}
    for section, keyword in _SECTION_MARKERS:
        if _strip_diacritics(keyword).lower() in stripped:
            found[section] = True
    return found


def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _keyword_to_pattern(keyword: str) -> str:
    """Convert a Vietnamese keyword into a flexible whitespace, case-insensitive regex."""
    normalized = _strip_diacritics(keyword).lower()
    return r"\s*".join(re.escape(c) for c in normalized)


def _extract_section(
    lines: list[str],
    patterns: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Extract line items matching *patterns* from *lines*.

    Looks for the label pattern on a line, then scans the same line and up
    to 3 subsequent lines for numerical values.  Vietnamese diacritics are
    stripped for fuzzy matching, and lines containing year values (1900-2100)
    are skipped when collecting values.
    """
    # Pre-compute diacritic-stripped lines once
    stripped_lines = [_strip_diacritics(line).lower() for line in lines]

    results: list[dict[str, Any]] = []
    for label, keyword in patterns:
        pattern = _keyword_to_pattern(keyword)
        values: list[float] = []
        for i, line in enumerate(lines):
            if re.search(pattern, stripped_lines[i]):
                # Try this line first
                num = _parse_number(line)
                if num is not None:
                    if not (1900 <= num <= 2100):  # skip year values
                        values.append(num)
                else:
                    # Try next few lines (label on its own line, numbers below)
                    for j in range(i + 1, min(i + 4, len(lines))):
                        nums = _parse_line_numbers(lines[j])
                        if nums:
                            # Filter out year-looking values
                            nums = [n for n in nums if not (1900 <= n <= 2100)]
                            if nums:
                                values.extend(nums)
                                break
        if values:
            results.append({
                "label": label,
                "values": values,
            })
    return results


def _parse_line_numbers(line: str) -> list[float]:
    """Extract all numbers from a line that may contain space-separated values."""
    cleaned = line.replace(",", "").replace("–", "-")
    nums: list[float] = []
    for token in cleaned.split():
        try:
            nums.append(float(token))
        except ValueError:
            pass
    return nums


def compute_derived_metrics(financials: dict[str, Any]) -> dict[str, Any]:
    """Derive financial metrics from extracted line items.

    Returns:
        - revenue_growth: list of YoY growth rates
        - gross_margin: list of gross margins
        - operating_margin: list of operating margins
        - net_margin: list of net margins
        - fcf: list of free cash flows (OCF - CapEx)
        - current_ratio: list of current ratios (if data available)
        - debt_to_equity: list of D/E ratios (if data available)
    """
    is_items = {r["label"]: r["values"] for r in financials.get("income_statement", [])}
    bs_items = {r["label"]: r["values"] for r in financials.get("balance_sheet", [])}
    cf_items = {r["label"]: r["values"] for r in financials.get("cash_flow", [])}

    metrics: dict[str, Any] = {}

    # Gross margin
    rev = is_items.get("doanh_thu_thuan", [])
    gp = is_items.get("loi_nhuan_gop", [])
    if rev and gp:
        n = min(len(rev), len(gp))
        metrics["gross_margin"] = [gp[i] / rev[i] * 100 if rev[i] else 0 for i in range(n)]

    # Operating margin
    op = is_items.get("loi_nhuan_hdkd", [])
    if rev and op:
        n = min(len(rev), len(op))
        metrics["operating_margin"] = [op[i] / rev[i] * 100 if rev[i] else 0 for i in range(n)]

    # Net margin
    ni = is_items.get("loi_nhuan_sau_thue", [])
    if rev and ni:
        n = min(len(rev), len(ni))
        metrics["net_margin"] = [ni[i] / rev[i] * 100 if rev[i] else 0 for i in range(n)]

    # Revenue growth
    if len(rev) >= 2:
        growth = []
        for i in range(1, len(rev)):
            if rev[i - 1]:
                growth.append((rev[i] - rev[i - 1]) / rev[i - 1] * 100)
            else:
                growth.append(0)
        metrics["revenue_growth"] = growth

    # Free cash flow (OCF - CapEx)
    ocf = cf_items.get("luu_chuyen_tu_hdkd", [])
    capex = cf_items.get("chi_mua_tscd", [])
    if ocf and capex:
        n = min(len(ocf), len(capex))
        metrics["fcf"] = [ocf[i] - abs(capex[i]) if capex[i] else ocf[i] for i in range(n)]

    # Debt to equity
    debt = bs_items.get("no_vay", [])
    debt = debt or bs_items.get("no_vay_ngan_han", [])
    equity = bs_items.get("von_chu_so_huu", [])
    if debt and equity:
        n = min(len(debt), len(equity))
        metrics["debt_to_equity"] = [debt[i] / equity[i] if equity[i] else 0 for i in range(n)]

    return metrics
