# filename: paper_modules.py
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from math_parsing import parse_math
from prompt import agent_prompt


# ============================================================
# Figure 2 — Dataset Processing / Normalization
# ============================================================

class DatasetNormalizer:
    """
    Figure 2: transform heterogeneous samples into a unified (question, answer, raw_task) triple.
    Delegates formatting to prompt.agent_prompt templates.
    """

    def normalize(self, dataset_name: str, sample: Any) -> Tuple[str, Any, Any]:
        if dataset_name == "mmlu":
            # sample can be list or dict-like; original code expects positional
            question_raw = sample[0]
            a, b, c, d = sample[1], sample[2], sample[3], sample[4]
            answer = sample[5]
            raw_task = tuple(sample) if isinstance(sample, list) else tuple(sample.values())
            question = agent_prompt[dataset_name]['question'].format(question_raw, a, b, c, d)
            return question, answer, raw_task

        if dataset_name == "math":
            question_raw = sample["problem"]
            answer = parse_math(sample["solution"])
            question = agent_prompt[dataset_name]['question'].format(question_raw)
            raw_task = sample
            return question, answer, raw_task

        if dataset_name == "chess":
            question_raw = sample["input"]
            last_move = question_raw.split(" ")[-1]
            question = agent_prompt[dataset_name]['question'].format(question_raw, last_move)
            answer = sample["target"]
            raw_task = sample
            return question, answer, raw_task

        if dataset_name == "mquake":
            question_raw = sample['questions'][0]
            answer = [sample['answer']] + sample.get('answer_alias', [])
            raw_task = sample
            question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
            return question, answer, raw_task

        if dataset_name == "musique":
            question_raw = sample['question']
            answer = [sample['answer']] + sample.get('answer_aliases', [])
            raw_task = sample
            question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
            return question, answer, raw_task

        if dataset_name == "truthfulqa":
            question_raw = sample['question']
            targets = sample['mc1_targets']  # list of 0/1
            # Create (letter, text) options
            options = [(chr(97 + i), opt) for i, opt in enumerate(targets)]
            # Correct option(s) are where target == 1
            correct = [(chr(97 + i), targets[i]) for i in range(len(targets)) if targets[i] == 1]
            # NOTE: original pipeline expects answer to be list with (letter, something).
            # Here we keep the letter but store the index only is not useful; better store letter + option text.
            correct_letters = [chr(97 + i) for i in range(len(targets)) if targets[i] == 1]
            answers_txt = ', '.join([f"({chr(97+i).upper()}) {targets[i]}" for i in range(len(targets))])

            # If your dataset actually stores choice texts elsewhere, adapt here.
            # Many TruthfulQA dumps store "mc1_choices" / "mc1_targets" (targets aligned to those choices).
            # If you have "mc1_choices", prefer that:
            if "mc1_choices" in sample:
                choices = sample["mc1_choices"]
                answers_txt = ', '.join([f"({chr(97+i).upper()}) {choices[i]}" for i in range(len(choices))])
                correct = [(chr(97 + i), choices[i]) for i in range(len(choices)) if targets[i] == 1]
            else:
                # fallback: treat targets as answer strings (less ideal)
                correct = [(l, targets_idx) for l, targets_idx in zip(correct_letters, correct_letters)]

            raw_task = sample
            question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
            return question, correct, raw_task

        if dataset_name == "medmcqa":
            question_raw = sample['question']
            letters = ['a', 'b', 'c', 'd']
            choices = [sample['opa'], sample['opb'], sample['opc'], sample['opd']]
            answer = letters[sample['cop'] - 1]
            raw_task = sample
            answers_txt = ', '.join([f"({l.upper()}) {c}" for l, c in zip(letters, choices)])
            question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
            return question, answer, raw_task

        if dataset_name == "scalr":
            question_raw = sample['question']
            letters = ['a', 'b', 'c', 'd', 'e']
            choices = [sample['choice_0'], sample['choice_1'], sample['choice_2'], sample['choice_3'], sample['choice_4']]
            answer = letters[sample['answer']]
            raw_task = sample
            answers_txt = ', '.join([f"({l.upper()}) {c}" for l, c in zip(letters, choices)])
            question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
            return question, answer, raw_task

        raise ValueError(f"Dataset {dataset_name} not supported")


# ============================================================
# Model Backends — OpenAI / HF
# ============================================================

@dataclass
class HFGenerationConfig:
    max_new_tokens: int = 1000
    do_sample: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class OpenAIChatBackend:
    """OpenAI Chat Completions backend."""
    def __init__(self, client):
        self.client = client

    def generate(self, messages: List[Dict[str, str]], model_name: str, n: int = 1) -> str:
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            n=n
        )
        return completion.choices[0].message.content


class HFChatBackend:
    """HuggingFace chat-template backend."""
    def __init__(self, model, tokenizer, gen_cfg: Optional[HFGenerationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_cfg = gen_cfg or HFGenerationConfig()

    def generate(self, messages: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [self.tokenizer.eos_token_id]
        name = (self.model.name_or_path or "").lower()
        if ("llama" in name) or ("gpt" in name):
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot is not None:
                terminators.append(eot)

        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            eos_token_id=terminators,
            do_sample=self.gen_cfg.do_sample,
        )
        if self.gen_cfg.temperature is not None:
            gen_kwargs["temperature"] = self.gen_cfg.temperature
        if self.gen_cfg.top_p is not None:
            gen_kwargs["top_p"] = self.gen_cfg.top_p

        outputs = self.model.generate(input_ids, **gen_kwargs)
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)


def load_model_tokenizer(model_name: str, device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    return model, tokenizer


# ============================================================
# Reliability wrappers (retry-safe)
# ============================================================

def _retry_sleep(seconds: int = 20) -> None:
    time.sleep(seconds)


def query_model(client, agent_context, model_name="gpt-3.5-turbo-0125"):
    """
    Backward-compatible helper.
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=agent_context,
            n=1
        )
    except Exception as e:
        print(f"retrying due to an error: {e}")
        _retry_sleep(20)
        return query_model(client, agent_context, model_name=model_name)

    return completion.choices[0].message.content


def query_model_extra(
    client,
    agent_context,
    model_name="gpt-3.5-turbo-0125",
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    n_repetitions: int = 1
):
    """
    Used for Best-of-N sampling and logprob scoring with a judge.
    Returns the full completion object.
    """
    tlp = None if not logprobs else top_logprobs
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=agent_context,
            n=n_repetitions,
            logprobs=logprobs,
            top_logprobs=tlp,
            max_tokens=max_tokens
        )
    except Exception as e:
        print(f"retrying due to an error: {e}")
        _retry_sleep(20)
        return query_model_extra(
            client,
            agent_context,
            model_name=model_name,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n_repetitions=n_repetitions
        )
    return completion


def query_hf_model(model, tokenizer, agent_context):
    """
    Backward-compatible HF helper.
    """
    backend = HFChatBackend(model, tokenizer)
    return backend.generate(agent_context)


# ============================================================
# Backward-compatible API (old name)
# ============================================================

def parse_question_answer(dataset_name: str, sample: Any):
    """
    Backward-compatible wrapper around DatasetNormalizer.
    """
    return DatasetNormalizer().normalize(dataset_name, sample)
