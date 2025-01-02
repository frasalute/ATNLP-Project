import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SCANDataset
from transformer import Transformer
from tqdm import tqdm
from pathlib import Path
import numpy as np


def calculate_accuracy(pred, target, pad_idx):
    """Calculate token and sequence accuracy after removing padding (could decrease accuracy keeping the padding)."""
    batch_size = pred.size(0)

    # Pad sequences to the same length
    max_len = max(pred.size(1), target.size(1))
    if pred.size(1) < max_len:
        pad_size = (batch_size, max_len - pred.size(1))
        pred = torch.cat([pred, torch.full(pad_size, pad_idx).to(pred.device)], dim=1)
    elif target.size(1) < max_len:
        pad_size = (batch_size, max_len - target.size(1))
        target = torch.cat([target, torch.full(pad_size, pad_idx).to(target.device)], dim=1)

    # Calculate sequence accuracy
    pred_stripped = [seq[seq != pad_idx].tolist() for seq in pred]
    target_stripped = [seq[seq != pad_idx].tolist() for seq in target]
    seq_acc = np.mean([p == t for p, t in zip(pred_stripped, target_stripped)])

    # Reshape to 1D and calculate token accuracy
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    mask = target != pad_idx
    correct = (pred[mask] == target[mask]).float()
    token_acc = correct.mean().item()

    return token_acc, seq_acc



def evaluate(model, data_loader, criterion, pad_idx, device):
    """Evaluate the model on validation/test data."""
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            # Prepare input and output sequences
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass through the model
            output = model(src, tgt_input)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()

            # Decode predictions and calculate accuracies
            pred = output.argmax(dim=-1).view(tgt.size(0), -1)
            token_acc, seq_acc = calculate_accuracy(pred, tgt[:, 1:], pad_idx)

            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    # Calculate averages
    avg_loss = total_loss / len(data_loader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc



def beam_search_decode(model, src, max_len, start_symbol, end_symbol, beam_size, device, pad_idx):
    """Beam search decoding for autoregressive generation."""
    model.eval()
    src = src.to(device)

    batch_size = src.size(0)
    beams = [
        [(torch.ones(1, 1).fill_(start_symbol).long().to(device), 0)] for _ in range(batch_size)
    ]

    finished_sequences = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        all_candidates = [[] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for seq, score in beams[batch_idx]:
                out = model(src[batch_idx].unsqueeze(0), seq)  # Process one sequence
                prob = torch.log_softmax(out[:, -1], dim=-1)
                topk_prob, topk_idx = prob.topk(beam_size, dim=-1)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, topk_idx[0, i].unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + topk_prob[0, i].item()
                    all_candidates[batch_idx].append((new_seq, new_score))

            all_candidates[batch_idx] = sorted(all_candidates[batch_idx], key=lambda x: x[1], reverse=True)[:beam_size]
            beams[batch_idx] = []
            for seq, score in all_candidates[batch_idx]:
                if seq[0, -1] == end_symbol:
                    finished_sequences[batch_idx].append((seq, score))
                else:
                    beams[batch_idx].append((seq, score))

        if all(len(beams[b]) == 0 for b in range(batch_size)):
            break

    # Select the best sequence for each batch
    best_sequences = []
    for batch_idx in range(batch_size):
        if finished_sequences[batch_idx]:
            best_seq = max(finished_sequences[batch_idx], key=lambda x: x[1])
        else:
            best_seq = max(beams[batch_idx], key=lambda x: x[1])
        best_sequences.append(best_seq[0])

    # Pad sequences to the same length
    max_seq_len = max(seq.size(1) for seq in best_sequences)
    padded_sequences = [
        torch.cat([seq, torch.full((1, max_seq_len - seq.size(1)), pad_idx, device=seq.device)], dim=1)
        if seq.size(1) < max_seq_len else seq
        for seq in best_sequences
    ]

    return torch.cat(padded_sequences, dim=0)



def train(train_path, test_path, hyperparams, model_suffix, random_seed):
    """Train the Transformer model"""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=4)

    model = Transformer(
        src_vocab_size=train_dataset.src_vocab.vocab_size,
        tgt_vocab_size=train_dataset.tgt_vocab.vocab_size,
        src_pad_idx=train_dataset.src_vocab.special_tokens["<PAD>"],
        tgt_pad_idx=train_dataset.src_vocab.special_tokens["<PAD>"],
        emb_dim=hyperparams["emb_dim"],
        num_layers=hyperparams["n_layers"],
        num_heads=hyperparams["n_heads"],
        forward_dim=hyperparams["forward_dim"],
        dropout=hyperparams["dropout"],
    ).to(hyperparams["device"])

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tgt_vocab.special_tokens["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    best_acc = 0.0
    pad_idx = train_dataset.src_vocab.special_tokens["<PAD>"]
    bos_idx = train_dataset.src_vocab.special_tokens["<BOS>"]
    eos_idx = train_dataset.src_vocab.special_tokens["<EOS>"]

    print(f"Training for {hyperparams['epochs']} epochs")
    for epoch in range(hyperparams["epochs"]):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hyperparams['epochs']} [Train]")
        for batch in pbar:
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss, avg_token_acc, avg_seq_acc = evaluate(model, test_loader, criterion, pad_idx, hyperparams["device"])
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Token Accuracy: {avg_token_acc:.4f}")
        print(f"Sequence Accuracy: {avg_seq_acc:.4f}")

        if avg_seq_acc > best_acc:
            best_acc = avg_seq_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparams": hyperparams,
                    "accuracy": best_acc,
                },
                CHECKPOINT_DIR / f"best_model_{model_suffix}.pt",
            )

    print("\nFinal Evaluation with Beam Search Decode:")
    model.eval()
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            pred = beam_search_decode(
                model,
                src,
                max_len=tgt.size(1),
                start_symbol=bos_idx,
                end_symbol=eos_idx,
                beam_size=5,
                device=hyperparams["device"],
                pad_idx=pad_idx,  # Pass pad_idx for consistent padding
            )

            # Ensure predictions and targets are the same length
            pred = pred[:, :tgt.size(1)]  # Truncate if necessary
            token_acc, seq_acc = calculate_accuracy(pred, tgt[:, 1:], pad_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)


    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    print(f"Final Token Accuracy: {avg_token_acc:.4f}")
    print(f"Final Sequence Accuracy: {avg_seq_acc:.4f}")
    return model, avg_token_acc, avg_seq_acc
