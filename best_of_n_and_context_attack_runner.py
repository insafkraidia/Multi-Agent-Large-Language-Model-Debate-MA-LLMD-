# filename: best_of_n_and_context_attack_runner.py
import os
import re
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from dataloader import get_dataset
from commons import (
    query_model,
    parse_question_answer,
    query_hf_model,
    load_model_tokenizer,
    query_model_extra,
)
from prompt import agent_prompt, adversary_prompt, optim


# -----------------------------
# Utilities
# -----------------------------

def parse_boxed_math(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


# -----------------------------
# Paper-aligned modules
# -----------------------------

class DatasetNormalizer:
    """Figure 2: normalize dataset → (question, answer, raw_task)."""
    def normalize(self, dataset_name: str, sample: Dict[str, Any]) -> Tuple[str, Any, Dict[str, Any]]:
        return parse_question_answer(dataset_name, sample)


class AdversarialAnswerSelector:
    """Select incorrect target answer y_wrong."""
    @staticmethod
    def _random_chess_move() -> str:
        possible_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        possible_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        return random.choice(possible_letter) + str(random.choice(possible_number))

    def select_incorrect_answer(self, dataset_name: str, answer: Any, raw_task: Dict[str, Any]) -> str:
        if dataset_name == 'truthfulqa':
            letters = [chr(97 + i) for i in range(len(raw_task['mc1_targets']))]
            letters.remove(answer[0][0])
            return random.choice(letters)

        if dataset_name == 'mmlu':
            letters = ['a', 'b', 'c', 'd']
            letters.remove(answer[0][0].lower())
            return random.choice(letters)

        if dataset_name == 'chess':
            correct_moves = answer
            m = correct_moves[0]
            while m in correct_moves:
                m = self._random_chess_move()
            return m

        if dataset_name == 'medmcqa':
            letters = ['a', 'b', 'c', 'd']
            letters.remove(answer.lower())
            return random.choice(letters)

        if dataset_name == 'scalr':
            letters = ['a', 'b', 'c', 'd', 'e']
            letters.remove(answer.lower())
            return random.choice(letters)

        raise ValueError(f"Dataset {dataset_name} not supported")


class DebatePromptBuilder:
    """Build cooperative debate message (Algorithm 1 prompt composition)."""
    def build_cooperative_debate_turn(self, dataset_name: str, other_contexts: List[List[Dict[str, str]]], idx: int) -> Dict[str, str]:
        prefix = agent_prompt[dataset_name]['debate'][0]
        for ctx in other_contexts:
            if ctx[idx]["role"] == "user":
                assert ctx[idx + 1]["role"] == "assistant"
                agent_response = ctx[idx + 1]["content"]
            else:
                agent_response = ctx[idx]["content"]
            prefix += f"\n\n One agent solution: ```{agent_response}```"
        prefix += agent_prompt[dataset_name]['debate'][1]
        return {"role": "user", "content": prefix}


@dataclass
class AttackMode:
    """Figure 10 ablations: Best-of-N and/or Knowledge Injection."""
    best_of_n: bool
    knowledge_injection: bool

    def validate(self) -> None:
        if not self.best_of_n and not self.knowledge_injection:
            raise ValueError("Invalid mode: at least one of best_of_n or knowledge_injection must be enabled.")


class AdversarialDebatePromptBuilder:
    """
    Builds adversarial debate message for enhanced attacks.
    - best_of_n: inject 'selected argument'
    - knowledge_injection: inject provided context
    """
    def build(
        self,
        dataset_name: str,
        other_contexts: List[List[Dict[str, str]]],
        y_wrong: str,
        idx: int,
        mode: AttackMode,
        selected_argument: Optional[str],
        injected_context: Optional[str],
    ) -> Dict[str, str]:
        prefix = agent_prompt[dataset_name]['debate'][0]
        for ctx in other_contexts:
            agent_response = ctx[idx]["content"]
            prefix += f"\n\n One agent solution: ```{agent_response}```"

        base = adversary_prompt[dataset_name]['debate']

        if mode.best_of_n and not mode.knowledge_injection:
            if not selected_argument:
                raise ValueError("best_of_n enabled but selected_argument is None.")
            prefix += base
            prefix += " You are also provided with your argument to use.\n\n"
            prefix += f"Your Argument: {selected_argument}\n\n"
            prefix += f"Your answer: ({y_wrong.upper()})"
            return {"role": "user", "content": prefix}

        if (not mode.best_of_n) and mode.knowledge_injection:
            if injected_context is None:
                raise ValueError("knowledge_injection enabled but injected_context is None.")
            prefix += base
            prefix += " Use the given answer and the context for your response.\n"
            prefix += f"Your answer: ({y_wrong.upper()})\n\n"
            prefix += f"Context: {injected_context}"
            return {"role": "user", "content": prefix}

        if mode.best_of_n and mode.knowledge_injection:
            if not selected_argument:
                raise ValueError("best_of_n+knowledge_injection enabled but selected_argument is None.")
            if injected_context is None:
                raise ValueError("best_of_n+knowledge_injection enabled but injected_context is None.")
            prefix += base
            prefix += " Use the given answer, the argument and the context for your response.\n"
            prefix += f"Your answer: ({y_wrong.upper()})\n\n"
            prefix += f"Your Argument: {selected_argument}\n\n"
            prefix += f"Context: {injected_context}"
            return {"role": "user", "content": prefix}

        raise ValueError("Invalid mode state.")


class BestOfNArgumentGenerator:
    """
    Best-of-N: sample N candidate arguments at inference time (paper: argument optimization).
    """
    def sample_candidates(
        self,
        client: OpenAI,
        adv_model: str,
        question: str,
        y_wrong: str,
        other_agent_contexts: Optional[List[List[Dict[str, str]]]],
        n_candidates: int,
        idx: Optional[int],
    ) -> List[str]:
        agent_solutions = ""
        if other_agent_contexts:
            agent_solutions = "Other agents responses were:\n"
            for ctx in other_agent_contexts:
                if idx is not None and ctx[idx]["role"] == "user":
                    assert ctx[idx + 1]["role"] == "assistant"
                    agent_response = ctx[idx + 1]["content"]
                elif idx is not None:
                    agent_response = ctx[idx]["content"]
                else:
                    # init step: no idx
                    agent_response = ctx[-1]["content"]
                agent_solutions += f"\n\n One agent solution: ```{agent_response}```"

        messages = [
            {"role": "system", "content": optim['argument_generator']['system']},
            {"role": "user", "content": optim['argument_generator']['user'].format(
                question=question,
                agent_responses=agent_solutions,
                adv_answer=y_wrong.upper()
            )},
        ]

        completion = query_model_extra(client, messages, adv_model, n_repetitions=n_candidates)
        return [c.message.content for c in completion.choices]


class BestOfNArgumentSelector:
    """
    Select most persuasive candidate using a judge model (logprob-based).
    """
    def select(
        self,
        client: OpenAI,
        judge_model: str,
        question: str,
        y_wrong: str,
        arguments: List[str],
    ) -> Tuple[str, List[float]]:
        judge_prompt = optim['argument_selector']['judge']
        dummy = optim['argument_selector']['dummy']

        scores: List[float] = []
        for argument in arguments:
            messages = [{
                "role": "system",
                "content": judge_prompt.format(
                    question=question,
                    debater_1=argument,
                    debater_2=dummy.format(answer=y_wrong.upper())
                )
            }]

            if "gpt" not in judge_model.lower():
                raise ValueError("Only GPT judge supported in current implementation (logprobs required).")

            completion = query_model_extra(
                client,
                messages,
                judge_model,
                logprobs=True,
                top_logprobs=10,
                max_tokens=1
            )

            top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs

            # Score = logprob(model chooses "1" i.e., argument is more persuasive)
            best_logprob_1 = -100.0
            for lp in top_logprobs:
                tok = lp.token.lower()
                if tok == '1' or tok == '(1':
                    best_logprob_1 = max(best_logprob_1, lp.logprob)

            scores.append(best_logprob_1)

        best_idx = scores.index(max(scores))
        return arguments[best_idx], scores


class ModelBackend:
    """Unifies OpenAI vs HF generation for group/adversary."""
    def __init__(self, group_model: str, adv_model: str):
        self.group_model_name = group_model
        self.adv_model_name = adv_model

        self.client: Optional[OpenAI] = None
        self.group_model = self.group_tokenizer = None
        self.adv_model = self.adv_tokenizer = None

        self._init()

    @staticmethod
    def _is_openai(name: str) -> bool:
        return "gpt" in name.lower()

    @staticmethod
    def _is_hf(name: str) -> bool:
        n = name.lower()
        return ("mistral" in n) or ("llama" in n) or ("qwen" in n) or ("yi" in n)

    def _init(self) -> None:
        if self._is_openai(self.group_model_name) or self._is_openai(self.adv_model_name):
            self.client = OpenAI()

        if self._is_hf(self.group_model_name):
            self.group_model, self.group_tokenizer = load_model_tokenizer(self.group_model_name)

        if self._is_hf(self.adv_model_name):
            self.adv_model, self.adv_tokenizer = load_model_tokenizer(self.adv_model_name)

        if (not self._is_openai(self.group_model_name)) and (not self._is_hf(self.group_model_name)):
            raise ValueError(f"Group model not supported: {self.group_model_name}")
        if (not self._is_openai(self.adv_model_name)) and (not self._is_hf(self.adv_model_name)):
            raise ValueError(f"Adversary model not supported: {self.adv_model_name}")

    def generate(self, role: str, messages: List[Dict[str, str]]) -> str:
        if role == "group":
            if self._is_hf(self.group_model_name):
                return query_hf_model(self.group_model, self.group_tokenizer, messages)
            return query_model(self.client, messages, self.group_model_name)

        if role == "adversary":
            if self._is_hf(self.adv_model_name):
                return query_hf_model(self.adv_model, self.adv_tokenizer, messages)
            return query_model(self.client, messages, self.adv_model_name)

        raise ValueError(f"Unknown role: {role}")


# -----------------------------
# Runner (Algorithm 1 + Figure 10 modes)
# -----------------------------

@dataclass
class AttackConfig:
    dataset: str
    input_file: Optional[str]
    n_samples: int
    n_agents: int
    n_rounds: int
    n_reps: int
    output_dir: str
    n_adversaries: int
    group_model: str
    adv_model: str
    judge_model: str
    gpus: str
    run_optim: bool
    run_context: bool
    n_arguments: int


class BestOfNContextAttackRunner:
    """
    Enhanced adversarial settings (Figure 10):
    - Best-of-N argument optimization
    - Knowledge/context injection
    """
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        assert self.cfg.n_adversaries <= self.cfg.n_agents

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpus

        self.normalizer = DatasetNormalizer()
        self.adv_selector = AdversarialAnswerSelector()
        self.coop_prompt_builder = DebatePromptBuilder()
        self.adv_prompt_builder = AdversarialDebatePromptBuilder()

        self.backend = ModelBackend(cfg.group_model, cfg.adv_model)
        self.client = self.backend.client  # needed for query_model_extra

        self.best_of_n_generator = BestOfNArgumentGenerator()
        self.best_of_n_selector = BestOfNArgumentSelector()

        self.mode = AttackMode(best_of_n=cfg.run_optim, knowledge_injection=cfg.run_context)
        self.mode.validate()

        self.out_dir = self._prepare_out_dir()

    def _prepare_out_dir(self) -> Path:
        def short(m: str) -> str:
            return m.split('/')[-1] if '/' in m else m

        suffix = ""
        if self.cfg.run_optim:
            suffix += "_optim"
        if self.cfg.run_context:
            suffix += "_context"

        out_dir = Path(
            self.cfg.output_dir,
            self.cfg.dataset,
            f"TRIAL_adv_plus{suffix}_{self.cfg.n_samples}_{self.cfg.n_agents}_{self.cfg.n_rounds}_{self.cfg.n_adversaries}"
            f"-{short(self.cfg.group_model)}-{short(self.cfg.adv_model)}-{short(self.cfg.judge_model)}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if self.cfg.input_file:
            with open(self.cfg.input_file, "r") as f:
                return [json.loads(line) for line in f]
        return get_dataset(dataset_name=self.cfg.dataset, n_samples=self.cfg.n_samples, context=self.cfg.run_context)

    @staticmethod
    def _assistant(text: str) -> Dict[str, str]:
        return {"role": "assistant", "content": text}

    def _init_contexts(self, question: str, y_wrong: str, question_context: Optional[str]) -> List[List[Dict[str, str]]]:
        contexts: List[List[Dict[str, str]]] = []
        for agent_id in range(self.cfg.n_agents):
            if agent_id < self.cfg.n_adversaries:
                selected_argument = None
                if self.mode.best_of_n:
                    # init: no other contexts
                    candidates = self.best_of_n_generator.sample_candidates(
                        client=self.client,
                        adv_model=self.cfg.adv_model,
                        question=question,
                        y_wrong=y_wrong,
                        other_agent_contexts=None,
                        n_candidates=self.cfg.n_arguments,
                        idx=None
                    )
                    selected_argument, _scores = self.best_of_n_selector.select(
                        client=self.client,
                        judge_model=self.cfg.judge_model,
                        question=question,
                        y_wrong=y_wrong,
                        arguments=candidates
                    )

                # Build initial adversary prompt based on mode and templates
                if self.mode.best_of_n and not self.mode.knowledge_injection:
                    adv_prompt_text = optim['adversary']["init_optim"].format(
                        question=question,
                        adv_answer=y_wrong.upper(),
                        adv_argument=selected_argument
                    )
                elif (not self.mode.best_of_n) and self.mode.knowledge_injection:
                    adv_prompt_text = optim['adversary']["init_context"].format(
                        question=question,
                        adv_answer=y_wrong.upper(),
                        context=question_context
                    )
                else:
                    adv_prompt_text = optim['adversary']["init_optim_context"].format(
                        question=question,
                        adv_answer=y_wrong.upper(),
                        adv_argument=selected_argument,
                        context=question_context
                    )

                contexts.append([
                    {"role": "system", "content": adversary_prompt['system']},
                    {"role": "user", "content": adv_prompt_text},
                ])
            else:
                contexts.append([{"role": "user", "content": question}])

        return contexts

    def run(self) -> None:
        dataset = self._load_dataset()

        suffix = ""
        if self.cfg.run_optim:
            suffix += "_optim"
        if self.cfg.run_context:
            suffix += "_context"

        for rep in range(self.cfg.n_reps):
            print(f"Rep {rep}/{self.cfg.n_reps}")

            def short(m: str) -> str:
                return m.split('/')[-1] if '/' in m else m

            fname = (
                f"adv_plus{suffix}_{self.cfg.dataset}_{self.cfg.n_samples}_{self.cfg.n_agents}_"
                f"{self.cfg.n_rounds}_{self.cfg.n_adversaries}_{rep}"
                f"-{short(self.cfg.group_model)}-{short(self.cfg.adv_model)}-{short(self.cfg.judge_model)}.jsonl"
            )

            with open(self.out_dir / fname, "w") as f_out:
                for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                    if self.cfg.input_file and isinstance(sample, dict) and "raw_task" in sample:
                        sample = sample["raw_task"]

                    question, answer, raw_task = self.normalizer.normalize(self.cfg.dataset, sample)
                    question_context = raw_task.get("context")

                    y_wrong = self.adv_selector.select_incorrect_answer(self.cfg.dataset, answer, raw_task)
                    agent_contexts = self._init_contexts(question, y_wrong, question_context)

                    for r in range(self.cfg.n_rounds):
                        for agent_id, ctx in enumerate(agent_contexts):
                            is_adv = agent_id < self.cfg.n_adversaries
                            role = "adversary" if is_adv else "group"

                            if r != 0:
                                other = agent_contexts[:agent_id] + agent_contexts[agent_id + 1:]
                                idx = 2 * r - 1

                                if is_adv:
                                    selected_argument = None
                                    if self.mode.best_of_n:
                                        candidates = self.best_of_n_generator.sample_candidates(
                                            client=self.client,
                                            adv_model=self.cfg.adv_model,
                                            question=question,
                                            y_wrong=y_wrong,
                                            other_agent_contexts=other,
                                            n_candidates=self.cfg.n_arguments,
                                            idx=idx
                                        )
                                        selected_argument, _scores = self.best_of_n_selector.select(
                                            client=self.client,
                                            judge_model=self.cfg.judge_model,
                                            question=question,
                                            y_wrong=y_wrong,
                                            arguments=candidates
                                        )

                                    msg = self.adv_prompt_builder.build(
                                        dataset_name=self.cfg.dataset,
                                        other_contexts=other,
                                        y_wrong=y_wrong,
                                        idx=idx,
                                        mode=self.mode,
                                        selected_argument=selected_argument,
                                        injected_context=question_context
                                    )
                                else:
                                    msg = self.coop_prompt_builder.build_cooperative_debate_turn(
                                        dataset_name=self.cfg.dataset,
                                        other_contexts=other,
                                        idx=idx
                                    )

                                ctx.append(msg)

                            completion = self.backend.generate(role=role, messages=ctx)
                            ctx.append(self._assistant(completion))

                    record = {
                        "id": i,
                        "question": question,
                        "answer": answer,
                        "raw_task": raw_task,
                        "y_wrong": y_wrong,
                        "mode": {"best_of_n": self.mode.best_of_n, "knowledge_injection": self.mode.knowledge_injection},
                        "agent_responses": agent_contexts,
                    }
                    f_out.write(json.dumps(record) + "\n")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="truthfulqa",
                   choices=["mmlu", "chess", "math", "mquake", "musique", "truthfulqa", "medmcqa", "scalr"])
    p.add_argument("--input_file", type=str, default=None, required=False)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--n_agents", type=int, default=3)
    p.add_argument("--n_rounds", type=int, default=3)
    p.add_argument("--n_reps", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="results/")
    p.add_argument("--n_adversaries", type=int, default=1)
    p.add_argument("--group_model", type=str, default="gpt-4o")
    p.add_argument("--adv_model", type=str, default="gpt-4o")
    p.add_argument("--gpus", type=str, default="0")

    # Enhanced attack modes (Figure 10)
    p.add_argument("--run_optim", action="store_true")
    p.add_argument("--run_context", action="store_true")
    p.add_argument("--n_arguments", type=int, default=10)
    p.add_argument("--judge_model", type=str, default="gpt-4o")
    return p


def main():
    args = build_parser().parse_args()
    cfg = AttackConfig(
        dataset=args.dataset,
        input_file=args.input_file,
        n_samples=args.n_samples,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_reps=args.n_reps,
        output_dir=args.output_dir,
        n_adversaries=args.n_adversaries,
        group_model=args.group_model,
        adv_model=args.adv_model,
        judge_model=args.judge_model,
        gpus=args.gpus,
        run_optim=args.run_optim,
        run_context=args.run_context,
        n_arguments=args.n_arguments,
    )
    runner = BestOfNContextAttackRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
