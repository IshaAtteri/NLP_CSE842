import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def get_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data, {"Content-Type": "application/json"} if data else {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def generate(config, profile, messages):
    exe, model = Path(config["server_exe"]), Path(profile["model_path"])
    if not exe.is_file() or not model.is_file():
        raise FileNotFoundError(f"Server or model missing: {exe}, {model}")
    log_path = ROOT / ".local/llama-server.log"
    log_path.parent.mkdir(exist_ok=True)
    base = config["base_url"].rstrip("/")
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [str(exe), "-m", str(model), "--ctx-size", str(config["context_tokens"]),
             *config["common_server_args"], *profile["server_args"]],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            for _ in range(300):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited; see {log_path}")
                try:
                    model_id = get_json(base + "/models", timeout=2)["data"][0]["id"]
                    break
                except (OSError, TimeoutError, RuntimeError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"Server did not start; see {log_path}")

            started = time.perf_counter()
            result = get_json(base + "/chat/completions", {
                "model": model_id, "messages": messages,
                "temperature": config["temperature"],
                "max_tokens": config["max_output_tokens"],
            }, timeout=3600)
            elapsed = time.perf_counter() - started
        finally:
            server.terminate()
            server.wait()

    tokens = result.get("usage", {}).get("completion_tokens")
    print(f"{tokens} output tokens in {elapsed:.1f}s" if tokens is not None
          else f"Server response in {elapsed:.1f}s", file=sys.stderr)
    return result["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("--question-id", default="q_0000")
    parser.add_argument("--passages", type=Path, default=Path("data/processed/multihoprag_passages.jsonl"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/answer_models.json")
    parser.add_argument("--profile")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = config["profiles"][args.profile or config["default_profile"]]
    passages = {row["passage_id"]: row for row in map(json.loads, args.passages.open(encoding="utf-8"))}
    question = next((row for row in map(json.loads, args.retrieval.open(encoding="utf-8"))
                     if row["question_id"] == args.question_id), None)
    if question is None:
        raise ValueError(f"Question {args.question_id} not found")
    top_k = args.top_k or config["top_k"]
    evidence = "\n\n".join(
        f"[{i}] {passages[result['passage_id']]['text']}"
        for i, result in enumerate(question["results"][:top_k], 1)
    )
    messages = [
        {"role": "system", "content": "Answer briefly using only the evidence. Cite passage numbers. If evidence is insufficient, say so."},
        {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question['query']}"},
    ]
    print(generate(config, profile, messages))


if __name__ == "__main__":
    main()
