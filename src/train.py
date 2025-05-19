import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import data
import model
import os

parser = argparse.ArgumentParser(description="PyTorch ptb Language Model")
parser.add_argument("--epochs", type=int, default=40, help="upper epoch limit")
parser.add_argument(
    "--train_batch_size", type=int, default=20, metavar="N", help="batch size"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=10, metavar="N", help="eval batch size"
)
parser.add_argument("--max_sql", type=int, default=256, help="sequence length")
parser.add_argument("--seed", type=int, default=1234, help="set random seed")
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--emb_dim", type=int, default=256)
parser.add_argument(
    "--hidden_size", type=int, default=256, help="hidden size for RNN/LSTM models"
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="dropout applied to layers"
)
parser.add_argument(
    "--model_type",
    type=str,
    default="rnn",
    choices=["transformer", "rnn", "lstm"],
    help="type of model to use",
)
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument("--use_pe", action="store_true")
parser.add_argument(
    "--patience", type=int, default=5, help="patience for early stopping"
)
parser.add_argument(
    "--save", type=str, default="best_model.pt", help="path to save the best model"
)
parser.add_argument("--cuda", action="store_true", help="use CUDA device")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id used")

args = parser.parse_args()

torch.manual_seed(args.seed)

use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {"train": train_batch_size, "valid": eval_batch_size}
data_loader = data.Corpus("data/ptb", batch_size, args.max_sql)

if args.model_type == "transformer":
    model = model.LMModel_transformer(
        nvoc=len(data_loader.vocabulary),
        num_layers=args.num_layers,
        dim=args.emb_dim,
        nhead=args.num_heads,
    )
elif args.model_type == "rnn":
    model = model.LMModel_RNN(
        nvoc=len(data_loader.vocabulary),
        num_layers=args.num_layers,
        dim=args.emb_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
elif args.model_type == "lstm":
    model = model.LMModel_LSTM(
        nvoc=len(data_loader.vocabulary),
        num_layers=args.num_layers,
        dim=args.emb_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
else:
    raise ValueError("Invalid model type")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=2, factor=0.5
)

log_dir = os.path.join(
    "runs", f"{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
writer = SummaryWriter(log_dir)

criterion = nn.CrossEntropyLoss()


def evaluate():
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            if args.model_type in ["rnn", "lstm"]:
                decode, _ = model(data)
            else:
                decode = model(data)
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    avg_loss_value = avg_loss.item() / idx
    print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss_value), avg_loss_value


def train():
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        if args.model_type in ["rnn", "lstm"]:
            decode, _ = model(data)
        else:
            decode = model(data)
        optimizer.zero_grad()
        loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if (idx + 1) % 50 == 0:
            print(f"The loss is {loss}")
        idx += 1
        avg_loss += loss
    avg_loss_value = avg_loss.item() / idx
    return math.exp(avg_loss_value), avg_loss_value


train_perplexity = []
valid_perplexity = []
train_loss_history = []
valid_loss_history = []
best_valid_ppl = float("inf")
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    print(f"开始训练 epoch {epoch}")

    train_ppl, train_loss = train()
    train_perplexity.append(train_ppl)
    train_loss_history.append(train_loss)

    valid_ppl, valid_loss = evaluate()
    valid_perplexity.append(valid_ppl)
    valid_loss_history.append(valid_loss)

    writer.add_scalar("Perplexity/train", train_ppl, epoch)
    writer.add_scalar("Perplexity/valid", valid_ppl, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/valid", valid_loss, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

    scheduler.step(valid_ppl)

    if valid_ppl < best_valid_ppl:
        best_valid_ppl = valid_ppl
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_ppl": train_ppl,
                "valid_ppl": valid_ppl,
            },
            f"{args.model_type}_{args.save}",
        )
        print(f"保存最佳模型，验证集困惑度: {valid_ppl:.2f}")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= args.patience:
        print(f"早停! {args.patience} 个epoch内验证集困惑度没有提升.")
        break

    print("-" * 89)
    print(
        f"| epoch {epoch:3d} | 用时 {time.time() - epoch_start_time:5.2f}s | "
        f"训练困惑度 {train_ppl:.2f} | 验证困惑度 {valid_ppl:.2f} | 训练loss {train_loss:.4f} | 验证loss {valid_loss:.4f}"
    )
    print("-" * 89)


writer.close()
