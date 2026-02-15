# metric_asc_sae_math_equivalence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# Paper technique naming:
#   ASC = Answer String Canonicalization
#   SAE = String-based Answer Equivalence (post-ASC)
# ============================================================

@dataclass(frozen=True)
class ASCRules:
    """Configuration for ASC (kept minimal; extend only if needed)."""
    convert_half_to_frac: bool = True


def _fix_fracs(s: str) -> str:
    """
    Normalize malformed \\frac patterns:
      - \\frac12 -> \\frac{1}{2}
      - \\frac1b -> \\frac{1}{b}
    """
    parts = s.split("\\frac")
    if len(parts) == 1:
        return s

    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if not part:
            # rare edge-case: string ends with "\frac"
            continue

        if part[0] == "{":
            out += part
            continue

        # Expect at least two characters: numerator + denominator starter
        if len(part) < 2:
            return s  # fall back to original (matches your previous behavior)

        a, b = part[0], part[1]
        if b != "{":
            tail = part[2:] if len(part) > 2 else ""
            out += f"{{{a}}}{{{b}}}{tail}"
        else:
            tail = part[2:] if len(part) > 2 else ""
            out += f"{{{a}}}{b}{tail}"

    return out


def _fix_a_slash_b(s: str) -> str:
    """
    If string is a simple integer fraction like "3/4", convert to LaTeX frac.
    """
    if s.count("/") != 1:
        return s

    a_str, b_str = s.split("/", 1)
    try:
        a = int(a_str)
        b = int(b_str)
        if s != f"{a}/{b}":
            return s
        return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        return s


def _remove_right_units(s: str) -> str:
    """
    Removes units expressed as: <expr>\\text{ <unit> } (right side only).
    """
    token = "\\text{ "
    if token not in s:
        return s

    # Your original assumed exactly two splits; here we just take prefix safely.
    return s.split(token, 1)[0]


def _fix_sqrt(s: str) -> str:
    """
    Normalize \\sqrt3 -> \\sqrt{3}
    """
    if "\\sqrt" not in s:
        return s

    parts = s.split("\\sqrt")
    out = parts[0]
    for part in parts[1:]:
        if not part:
            out += "\\sqrt"
            continue
        if part[0] != "{":
            out += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            out += "\\sqrt" + part
    return out


def _remove_text_tag(original: str) -> str:
    """
    Remove '\\text{' and a matching trailing '}' once (best-effort).
    """
    s = original.replace("\\text{", "")
    if s == original:
        return s

    # remove one '}' from the end-most occurrence (not necessarily last char)
    rev = s[::-1]
    if "}" in rev:
        rev = rev.replace("}", "", 1)
    return rev[::-1]


def canonicalize_math_answer(ans: str, rules: ASCRules = ASCRules()) -> str:
    """
    ASC: Answer String Canonicalization for math / LaTeX-like answers.
    Mirrors your original `_strip_string` behavior.
    """
    s = ans.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = _remove_text_tag(s)

    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")

    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")

    s = _remove_right_units(s)

    s = s.replace("\\%", "").replace("\%", "")

    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if not s:
        return s

    if s[0] == ".":
        s = "0" + s

    # Remove simple "k=" prefix style
    if s.count("=") == 1:
        lhs, rhs = s.split("=", 1)
        if len(lhs) <= 2:
            s = rhs

    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)

    if rules.convert_half_to_frac and s == "0.5":
        s = "\\frac{1}{2}"

    s = _fix_a_slash_b(s)
    return s.strip()


def are_answers_equivalent(
    pred: Optional[str],
    gold: Optional[str],
    *,
    verbose: bool = False,
    rules: ASCRules = ASCRules(),
) -> bool:
    """
    SAE: String-based Answer Equivalence after ASC.
    """
    if pred is None and gold is None:
        if verbose:
            print("WARNING: Both None")
        return True
    if pred is None or gold is None:
        return False

    try:
        p = canonicalize_math_answer(pred, rules=rules)
        g = canonicalize_math_answer(gold, rules=rules)
        if verbose:
            print(p, g)
        return p == g
    except Exception:
        # last-resort fallback (same as your old code)
        return pred == gold
