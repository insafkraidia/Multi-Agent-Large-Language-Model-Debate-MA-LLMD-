# metric_fas_boxed_extractor.py
from __future__ import annotations
from typing import Optional


# ============================================================
# Paper Technique:
#   FAS = Final Answer Selector
#   BAE = Boxed Answer Extractor
# ============================================================


def _find_last_boxed_block(text: str) -> Optional[str]:
    """
    Locate the last \\boxed{...} or \\fbox{...} block,
    returning the full block including the command.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    brace_depth = 0
    end_idx = None

    while i < len(text):
        if text[i] == "{":
            brace_depth += 1
        elif text[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end_idx = i
                break
        i += 1

    if end_idx is None:
        return None

    return text[idx:end_idx + 1]


def _strip_boxed_wrapper(boxed_block: str) -> Optional[str]:
    """
    Remove '\\boxed{' and trailing '}'.
    """
    if boxed_block is None:
        return None

    if boxed_block.startswith("\\boxed{") and boxed_block.endswith("}"):
        return boxed_block[len("\\boxed{"):-1]

    if boxed_block.startswith("\\fbox{") and boxed_block.endswith("}"):
        return boxed_block[len("\\fbox{"):-1]

    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    FAS (Final Answer Selector):
    Extracts the last boxed answer from model output.
    """
    block = _find_last_boxed_block(text)
    return _strip_boxed_wrapper(block)
