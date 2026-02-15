# runner_cmd_clean_debate.py
import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI

from dataloader import get_dataset
from commons import (
    query_model,
    parse_question_answer,
    query_hf_model,
    load_model_tokenizer,
)
from prompt import agent_prompt


# ============================================================
# Paper technique: CMD (Clean Multi-Agent Debate)
# ============================================================

def build_cmd_debate_turn_prompt(dataset_name: str, peer_contexts: List[List[Dict[str, str]]], idx: int) -> Dict[str, str]:
    """
    Paper mapping:
      - CMD: Each agent receives the other agents' most recent solutions and continues debating.
    idx: index of the assistant turn to read from each peer context.
         For your chat format: assistant turns are at positions 2*round (after adding debate user msg).
         In your previous implementation you used (2*round - 1) for the user msg index.
         Here we always read the assistant content at idx from peer contexts.
    """
    prefix = agent_prompt[dataset_name]["debate"][0]

    for peer in peer_contexts:
        # peer[idx] expected to be assistant
        peer_solution = peer[idx]["content"]
        prefix += f"\n\n One agent solution: ```{peer_solution}```"

    prefix += agent_prompt[dataset_name]["debate"][1]
    return {"role": "user", "content": prefix}


def as_assistant_message(text: str) -> Dict[str, str]:
    return {"role": "assistant", "content": text}


# ============================================================
# Backend selection (paper-agnostic utility)
# ============================================================

class ModelBackend:
    def __init__(self, model_name: str, gpus: str = "0"):
        self.model_name = model_name
        self.gpus = gpus
        self.client: Optional[OpenAI] = None
        self.hf_model = None
        self.hf_tokenizer = None

        self._init_backend()

    def _init_backend(self) -> None:
        if "gpt" in self.model_name:
            self.client = OpenAI()
            return

        if any(x in self.model_name for x in ["mistral", "llama", "Yi", "Qwen"]):
            self.hf_model, self.hf_tokenizer = load_model_tokenizer(self.model_name)
            return

        raise ValueError(f"Model {self.model_name} not supported")

    def generate(self, context: List[Dict[str, str]]) -> str:
        if self.client is not None:
            return query_model(self.client, context, model_name=self.model_name)

        return query_hf_model(self.hf_model, self.hf_tokenizer, context)


# ============================================================
# Main CMD runner
# ============================================================

def run_cmd(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model_tag = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name

    # paper naming: CMD
    out_dir = Path(args.output_dir, args.dataset, f"CMD_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{model_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    if args.input_file:
        with open(args.input_file, "r") as f:
            dataset = [json.loads(line) for line in f]
    else:
        dataset = get_dataset(dataset_name=args.dataset, n_samples=args.n_samples)

    backend = ModelBackend(args.model_name, gpus=args.gpus)

    for rep in range(args.n_reps):
        fname = f"CMD_{args.dataset}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_rep{rep}_{model_tag}.jsonl"
        out_path = out_dir / fname

        with open(out_path, "w") as f:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"CMD rep {rep}"):
                if args.input_file:
                    sample = sample["raw_task"]

                question, answer, raw_task = parse_question_answer(args.dataset, sample)

                # initialize each agent with the same user question
                agent_contexts: List[List[Dict[str, str]]] = [
                    [{"role": "user", "content": question}] for _ in range(args.n_agents)
                ]

                for r in range(args.n_rounds):
                    for a_idx, a_ctx in enumerate(agent_contexts):
                        # from round 1 onward, add the debate prompt containing peers' previous assistant outputs
                        if r != 0:
                            peers = agent_contexts[:a_idx] + agent_contexts[a_idx + 1 :]

                            # previous assistant message index for peers:
                            # At end of round (r-1), each peer appended assistant => index = 2*(r-1)+1? (depends)
                            # In your original layout: [user question] then each round adds:
                            #   if r!=0: add user debate msg, then assistant
                            # Round 0: assistant at index 1
                            # Round 1: debate user at index 2, assistant at index 3
                            # Round 2: debate user at index 4, assistant at index 5
                            # => assistant index = 2*r + 1
                            prev_assistant_idx = 2 * (r - 1) + 1

                            debate_msg = build_cmd_debate_turn_prompt(args.dataset, peers, prev_assistant_idx)
                            a_ctx.append(debate_msg)

                        completion = backend.generate(a_ctx)
                        a_ctx.append(as_assistant_message(completion))

                if args.verbose:
                    print("question:", question)
                    print("answer:", answer)

                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "question": question,
                            "answer": answer,
                            "raw_task": raw_task,
                            "agent_responses": agent_contexts,
                            "technique": "CMD",
                            "model": args.model_name,
                            "n_agents": args.n_agents,
                            "n_rounds": args.n_rounds,
                            "rep": rep,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="truthfulqa",
                    choices=["mmlu", "chess", "math", "mquake", "musique", "truthfulqa", "medmcqa", "scalr"])
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--input_file", type=str, default=None)

    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_rounds", type=int, default=3)
    ap.add_argument("--n_reps", type=int, default=5)

    ap.add_argument("--output_dir", type=str, default="results/")
    ap.add_argument("--model_name", type=str, default="gpt-4o")

    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    run_cmd(args)
