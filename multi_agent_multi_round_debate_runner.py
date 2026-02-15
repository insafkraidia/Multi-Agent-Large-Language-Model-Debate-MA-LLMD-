# filename: multi_agent_multi_round_debate_runner.py
import os
import re
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from dataloader import get_dataset
from commons import (
    query_model,
    parse_question_answer,
    query_hf_model,
    load_model_tokenizer,
)
from prompt import agent_prompt, adversary_prompt


# -----------------------------
# Utilities (dataset-specific parsing helpers)
# -----------------------------

def parse_boxed_math(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content. Used for math datasets if needed."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


# -----------------------------
# Paper-aligned technique modules
# -----------------------------

class DatasetNormalizer:
    """
    Figure 2: Dataset processing / normalization.
    For now, delegates to commons.parse_question_answer to keep behavior stable.
    """
    def normalize(self, dataset_name: str, sample: Dict[str, Any]) -> Tuple[str, Any, Dict[str, Any]]:
        question, answer, raw_task = parse_question_answer(dataset_name, sample)
        return question, answer, raw_task


class AdversarialAnswerSelector:
    """
    Threat Model: adversary selects an incorrect target answer y_wrong.
    """
    @staticmethod
    def _random_chess_move() -> str:
        possible_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        possible_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return random.choice(possible_letter) + str(random.choice(possible_number))

    def select_incorrect_answer(self, dataset_name: str, answer: Any, raw_task: Dict[str, Any]) -> str:
        # Note: answer formats vary by dataset (some are list-like, some strings).
        if dataset_name == 'truthfulqa':
            answer_letters = [chr(97 + i) for i in range(len(raw_task['mc1_targets']))]
            correct = answer[0][0]  # e.g., "a"
            answer_letters.remove(correct)
            return random.choice(answer_letters)

        if dataset_name == 'mmlu':
            answer_letters = ['a', 'b', 'c', 'd']
            correct = answer[0][0].lower()
            answer_letters.remove(correct)
            return random.choice(answer_letters)

        if dataset_name == 'chess':
            correct_moves = answer  # list of valid moves
            random_move = self._random_chess_move()
            while random_move in correct_moves:
                random_move = self._random_chess_move()
            return random_move

        if dataset_name == 'medmcqa':
            answer_letters = ['a', 'b', 'c', 'd']
            correct = answer.lower()
            answer_letters.remove(correct)
            return random.choice(answer_letters)

        if dataset_name == 'scalr':
            answer_letters = ['a', 'b', 'c', 'd', 'e']
            correct = answer.lower()
            answer_letters.remove(correct)
            return random.choice(answer_letters)

        raise ValueError(f"Dataset {dataset_name} not supported")


class DebatePromptBuilder:
    """
    Algorithm 1, Step 12: Construct debate prompt prefix + other agents' answers + suffix.
    Also supports adversarial debate prompt that includes y_wrong instruction.
    """
    def build_cooperative_debate_turn(
        self,
        dataset_name: str,
        other_agent_contexts: List[List[Dict[str, str]]],
        idx: int
    ) -> Dict[str, str]:
        prefix = agent_prompt[dataset_name]['debate'][0]

        for agent_ctx in other_agent_contexts:
            # Handles system-prompt offset cases
            if agent_ctx[idx]["role"] == "user":
                assert agent_ctx[idx + 1]["role"] == "assistant"
                agent_response = agent_ctx[idx + 1]["content"]
            else:
                agent_response = agent_ctx[idx]["content"]

            prefix += f"\n\n One agent solution: ```{agent_response}```"

        prefix += agent_prompt[dataset_name]['debate'][1]
        return {"role": "user", "content": prefix}

    def build_adversarial_debate_turn(
        self,
        dataset_name: str,
        other_agent_contexts: List[List[Dict[str, str]]],
        y_wrong: str,
        idx: int
    ) -> Dict[str, str]:
        prefix = agent_prompt[dataset_name]['debate'][0]

        for agent_ctx in other_agent_contexts:
            agent_response_
