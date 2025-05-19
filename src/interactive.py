import torch
import random
import argparse
import model
import data

parser = argparse.ArgumentParser(
    description="Interactive Language Model Text Generation"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="rnn_best_model.pt",
    help="Path to model checkpoint",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="rnn",
    choices=["transformer", "rnn", "lstm"],
    help="Model type to use",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Generation temperature, higher means more random",
)
parser.add_argument(
    "--max_length", type=int, default=100, help="Maximum number of words to generate"
)
parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
parser.add_argument(
    "--data_path", type=str, default="data/ptb", help="Data path for loading vocabulary"
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
parser.add_argument("--no_cuda", action="store_true", help="Don't use CUDA")

args = parser.parse_args()

torch.manual_seed(1234)
random.seed(1234)

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
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[1])
    return torch.where(
        logits < min_values, torch.ones_like(logits) * -float("inf"), logits
    )


def generate_text(prompt, temp=None, length=None, k=None):
    temperature = temp if temp is not None else args.temperature
    max_length = length if length is not None else args.max_length
    top_k = k if k is not None else args.top_k

    input_ids = tokenize_prompt(prompt)
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

        for _ in range(max_length):
            logits = logits / temperature
            logits = top_k_logits(logits.unsqueeze(0), top_k).squeeze(0)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # if id_to_word[next_token.item()] == "<eos>":
            #     break

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

    return " ".join(generated_words)


def input_ids_to_words(ids):
    return [id_to_word[id] for id in ids]


print("\n=== Interactive Text Generation ===")
print("Enter a prompt and the model will generate text. Type 'exit' or 'quit' to exit.")
print("Or type 'config' to view/modify settings\n")

while True:
    try:
        prompt = input(">> Prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        if prompt.lower() == "config":
            print(f"\nCurrent settings:")
            print(f"  Model type: {args.model_type}")
            print(f"  Temperature: {args.temperature}")
            print(f"  Max length: {args.max_length}")
            print(f"  Top-K: {args.top_k}")

            change = input("Change settings? (y/n): ")
            if change.lower() == "y":
                try:
                    temp = float(
                        input(f"Temperature ({args.temperature}): ") or args.temperature
                    )
                    length = int(
                        input(f"Max length ({args.max_length}): ") or args.max_length
                    )
                    top_k = int(input(f"Top-K ({args.top_k}): ") or args.top_k)

                    args.temperature = temp
                    args.max_length = length
                    args.top_k = top_k

                    print("Settings updated!")
                except ValueError:
                    print("Invalid input, keeping current settings")
            continue

        if not prompt.strip():
            continue

        print("\nGenerating...\n")
        generated_text = generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")
        print("\n" + "-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nProgram terminated")
        break
    except Exception as e:
        print(f"Error: {e}")
