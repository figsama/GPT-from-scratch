"""
Text generation: load trained model and generate text
"""

import argparse
import torch
import math
import random
import sys
import model
import data

parser = argparse.ArgumentParser(description="Language Model Text Generation")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument(
    "--checkpoint", type=str, default="rnn_best_model.pt", help="Model checkpoint path"
)
parser.add_argument(
    "--model_type",
    type=str,
    default="rnn",
    choices=["transformer", "rnn", "lstm"],
    help="Model type",
)
parser.add_argument(
    "--prompt", type=str, default="the meaning of life", help="Prompt text"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Generation temperature, higher means more random",
)
parser.add_argument(
    "--max_length", type=int, default=50, help="Maximum number of words to generate"
)
parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
parser.add_argument(
    "--data_path", type=str, default="data/ptb", help="Data path for loading vocabulary"
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
parser.add_argument("--no_cuda", action="store_true", help="Don't use CUDA")
parser.add_argument(
    "--eval_file",
    type=str,
    default="data/valid.txt",
    help="Evaluate perplexity on a given text file (e.g., WikiText-2 valid.txt or test.txt)",
)
parser.add_argument(
    "--eval_batch_size", type=int, default=20, help="Batch size for evaluation"
)
parser.add_argument(
    "--eval_seq_len", type=int, default=256, help="Sequence length for evaluation"
)

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device("cpu" if args.no_cuda else f"cuda:{args.gpu_id}")
if not args.no_cuda:
    torch.cuda.set_device(args.gpu_id)

print(f"Loading vocabulary from: {args.data_path}")
dummy_batch_size = {"train": 1, "valid": 1}
corpus = data.Corpus(args.data_path, dummy_batch_size, 1)
vocab_size = len(corpus.vocabulary)
word_to_id = corpus.word_id
id_to_word = {v: k for k, v in word_to_id.items()}
print(f"Vocabulary size: {vocab_size}")

print(f"Loading {args.model_type} model from checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device)

if args.model_type == "transformer":
    model_instance = model.LMModel_transformer(nvoc=vocab_size)
elif args.model_type == "rnn":
    model_instance = model.LMModel_RNN(nvoc=vocab_size)
elif args.model_type == "lstm":
    model_instance = model.LMModel_LSTM(nvoc=vocab_size)
else:
    raise ValueError(f"Unsupported model type: {args.model_type}")

model_instance.load_state_dict(checkpoint["model_state_dict"])
model_instance.to(device)
model_instance.eval()


def tokenize_prompt(prompt):
    """Convert prompt text to token ID sequence"""
    words = prompt.strip().lower().split()
    ids = []
    for word in words:
        if word in word_to_id:
            ids.append(word_to_id[word])
        else:
            print(f"Warning: '{word}' not in vocabulary, skipped")
    if not ids:
        print("Warning: All words are out of vocabulary, using random word to start")
        ids = [random.randint(0, vocab_size - 1)]
    return ids


def top_k_logits(logits, k):
    """Apply top-k filtering"""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[1])
    return torch.where(
        logits < min_values, torch.ones_like(logits) * -float("inf"), logits
    )


def generate_text():
    input_ids = tokenize_prompt(args.prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)
    generated_words = []
    past = None

    with torch.no_grad():
        if args.model_type in ["rnn", "lstm"]:
            output, past = model_instance(input_tensor)
            logits = output[-1].squeeze(0)
        else:
            output = model_instance(input_tensor)
            logits = output[-1, 0, :]
        for _ in range(args.max_length):
            logits = logits / args.temperature
            logits = top_k_logits(logits.unsqueeze(0), args.top_k).squeeze(0)

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            if id_to_word[next_token.item()] == "<eos>":
                break
            generated_words.append(id_to_word[next_token.item()])

            if args.model_type in ["rnn", "lstm"]:
                next_token = next_token.view(1, 1)
                output, past = model_instance(next_token, past)
                logits = output.squeeze(0).squeeze(0)
            else:
                next_token = next_token.unsqueeze(1)
                input_tensor = torch.cat([input_tensor, next_token], dim=0)
                output = model_instance(input_tensor)
                logits = output[-1, 0, :]
    return " ".join(input_ids_to_words(input_ids)) + " " + " ".join(generated_words)


def input_ids_to_words(ids):
    return [id_to_word[id] for id in ids]


def evaluate_file_perplexity(eval_file, batch_size=20, seq_len=256):
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(reduction="sum")
    with open(eval_file, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        words = line.strip().split() + ["<eos>"]
        for word in words:
            if word in word_to_id:
                tokens.append(word_to_id[word])
            else:
                continue
    tokens = torch.tensor(tokens, dtype=torch.long)
    total_loss = 0.0
    total_tokens = 0
    model_instance.eval()
    with torch.no_grad():
        batch_num = tokens.size(0) // (batch_size * seq_len)
        usable_tokens = batch_num * batch_size * seq_len
        if usable_tokens == 0:
            return
        tokens = tokens[:usable_tokens]
        data = tokens.view(batch_size, -1).t().contiguous()
        for i in range(0, data.size(0) - 1, seq_len):
            seq_data = data[i : i + seq_len, :]
            seq_target = data[i + 1 : i + 1 + seq_len, :].reshape(-1)
            if (
                seq_data.size(0) != seq_len
                or seq_target.size(0) != batch_size * seq_len
            ):
                continue
            seq_data = seq_data.to(device)
            seq_target = seq_target.to(device)
            if args.model_type in ["rnn", "lstm"]:
                output, _ = model_instance(seq_data)
            else:
                output = model_instance(seq_data)
            loss = criterion(output.view(-1, vocab_size), seq_target)
            total_loss += loss.item()
            total_tokens += seq_target.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"\nFile: {eval_file}\nPerplexity: {ppl:.4f}\n")
    return ppl


print("\nInitial prompt:", args.prompt)
print("\nGenerated text:")
generated_text = generate_text()
print(generated_text)
print("\n")

if args.eval_file:
    evaluate_file_perplexity(
        args.eval_file, batch_size=args.eval_batch_size, seq_len=args.eval_seq_len
    )
    sys.exit(0)
