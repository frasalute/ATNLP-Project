import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SCANDataset
from transformer import Transformer
from tqdm import tqdm
from pathlib import Path
import numpy as np

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """Calculate cross-entropy loss, with optional label smoothing."""
    gold = gold.contiguous().reshape(-1)  # Ensure gold is contiguous and reshape

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.reshape(-1, 1), 1)  # Use reshape
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)(pred, gold)
    return loss

def beam_search_decode(model, src, max_len, start_symbol, end_symbol, beam_size, device):
    """Beam search decoding for autoregressive generation."""
    model.eval()
    src = src.to(device)

    batch_size = src.size(0)
    beams = [(torch.ones(batch_size, 1).fill_(start_symbol).long().to(device), 0)]  # (sequence, score)

    finished_sequences = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            out = model(src, seq)
            prob = torch.log_softmax(out[:, -1], dim=-1)  # Get probabilities for the next token
            topk_prob, topk_idx = prob.topk(beam_size, dim=-1)  # Get top-k tokens

            for i in range(beam_size):
                new_seq = torch.cat([seq, topk_idx[:, i].unsqueeze(1)], dim=-1)
                new_score = score + topk_prob[:, i].item()
                candidates.append((new_seq, new_score))

        # Sort candidates by score and keep the top beams
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        # Separate finished sequences
        beams = []
        for seq, score in candidates:
            if (seq[:, -1] == end_symbol).all():  # If the sequence ends with <EOS>
                finished_sequences.append((seq, score))
            else:
                beams.append((seq, score))

        if len(beams) == 0:
            break  # All beams are finished

    # Combine unfinished and finished sequences
    finished_sequences.extend(beams)

    # Return the sequence with the highest score
    best_seq, best_score = max(finished_sequences, key=lambda x: x[1])
    return best_seq

def calculate_accuracy(pred, target, pad_idx):
    """Calculate token and sequence accuracy"""
    batch_size = pred.size(0)

    max_len = max(pred.size(1), target.size(1))
    if pred.size(1) < max_len:
        pad_size = (batch_size, max_len - pred.size(1))
        pred = torch.cat([pred, torch.full(pad_size, pad_idx).to(pred.device)], dim=1)
    elif target.size(1) < max_len:
        pad_size = (batch_size, max_len - target.size(1))
        target = torch.cat([target, torch.full(pad_size, pad_idx).to(target.device)], dim=1)

    pred_lengths = (pred != pad_idx).sum(dim=1)
    target_lengths = (target != pad_idx).sum(dim=1)
    seq_acc = (pred_lengths == target_lengths).float().mean().item()

    pred = pred.contiguous().reshape(-1)
    target = target.contiguous().reshape(-1)

    mask = target != pad_idx
    correct = (pred[mask] == target[mask]).float()
    token_acc = correct.mean().item()

    return token_acc, seq_acc

def evaluate(model, data_loader, trg_pad_idx, device):
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            loss = cal_loss(output.contiguous().reshape(-1, output.size(-1)), 
                            tgt_output.contiguous().reshape(-1), trg_pad_idx)
            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            token_acc, seq_acc = calculate_accuracy(pred, tgt[:, 1:], trg_pad_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(data_loader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc

def train(train_path, test_path, hyperparams, model_suffix, random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False, 
                             num_workers=4, pin_memory=True)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * hyperparams["warmup_steps"] ** -1.5))

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

            loss = cal_loss(output.contiguous().reshape(-1, output.size(-1)), 
                            tgt_output.contiguous().reshape(-1), pad_idx, smoothing=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss, avg_token_acc, avg_seq_acc = evaluate(model, test_loader, pad_idx, hyperparams["device"])

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Token Accuracy: {avg_token_acc:.4f}")
        print(f"Sequence Accuracy: {avg_seq_acc:.4f}")

        if avg_seq_acc > best_acc:
            best_acc = avg_seq_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'hyperparams': hyperparams,
                'accuracy': best_acc,
            }, CHECKPOINT_DIR / f"best_model_{model_suffix}.pt")

    print("\nFinal Evaluation with Beam Search Decode:")
    model.eval()
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            src = batch["src"].to(hyperparams["device"])
            tgt = batch["tgt"].to(hyperparams["device"])

            pred = beam_search_decode(
                model, src, 
                max_len=tgt.size(1),
                start_symbol=bos_idx,
                end_symbol=eos_idx,
                beam_size=5,  # Set beam size
                device=hyperparams["device"]
            )

            token_acc, seq_acc = calculate_accuracy(pred[:, 1:], tgt[:, 1:], pad_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    print(f"Final Token Accuracy: {avg_token_acc:.4f}")
    print(f"Final Sequence Accuracy: {avg_seq_acc:.4f}")
    return model, avg_token_acc, avg_seq_acc
