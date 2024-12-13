import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from dataset import SCANDataset
from model.transformer import Transformer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

GRAD_CLIP = 1

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

        # Iterates over batches provided by the dataloader. tqdm: Displays a progress bar. 
    for batch in tqdm(dataloader, desc="Training"):

        # Extract Inputs and Targets - moves them to the specified device (GPU or CPU).
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        
        # Target Sequence Handling
        # Removes the last token of the target sequence <EOS>
        tgt_input = tgt[:, :-1]
        # Removes the first token of the target sequence
        tgt_output = tgt[:, 1:]

        # Clears gradients from the previous step to prevent accumulation
        optimizer.zero_grad()
        # Passes the source sequence (src) and the target input sequence (tgt_input) through the model.
        output = model(src, tgt_input)
        # Reshapes the model's outputs for compatibility with the loss function
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        # Compute Loss
        loss = criterion(output, tgt_output)
        # Backward Pass
        loss.backward()
        # Gradient Clipping maximum value of GRAD_CLIP (1 in this case) to prevent exploding gradients
        nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        # Updates the model’s parameters using the optimizer
        optimizer.step()
        
        # Accumulate Loss
        total_loss += loss.item()
    # Return Average Loss
    return total_loss / len(dataloader)

# evaluate the model on a validation set
def evaluate(model, dataloader, criterion, device):
    # Set the Model to Evaluation Mode
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        # Iterates through the dataloader to process each batch of data
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            # Prepare Target Sequences for Decoder
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward Pass Through the Model
            output = model(src, tgt_input)
            # Reshape Outputs
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            # Compute Loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    # Return Average Loss
    return total_loss / len(dataloader)

def calculate_accuracy(model, test_loader, dataset, device):
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in test_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            # Converts logits to token predictions by taking the index of the highest probability for each token.
            output = output.argmax(dim=-1)
            # store the predictions
            all_preds.extend(output.cpu().numpy().tolist())
            # store the real target
            all_targets.extend(tgt_output.cpu().numpy().tolist())

    # Filter and process predictions
    # Ensures predictions and targets have the same length and stops processing at the <EOS> token.
    filtered_preds = []
    filtered_targets = []
    for i in range(len(all_preds)):
        pred_seq = []
        for j in range(len(all_preds[i])):
            if all_preds[i][j] == tgt_eos_idx:
                break
            pred_seq.append(all_preds[i][j])
        filtered_preds.append(pred_seq)
        
        target_seq = []
        for j in range(len(pred_seq)):
            target_seq.append(all_targets[i][j])
        filtered_targets.append(target_seq)

    # Flattens the filtered predictions/targets into a single list of tokens
    flat_preds = [item for sublist in filtered_preds for item in sublist]
    flat_targets = [item for sublist in filtered_targets for item in sublist]

    # Compares flat_preds with flat_targets and calculate accuracy
    return accuracy_score(flat_targets, flat_preds)

# Created for Exp3
def calculate_sequence_accuracy(model, test_loader, dataset, device):
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    model.eval()
    
    # Initialize counters for correct sequences and total sequences
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            # Convert the model's logits to token predictions (indices) by taking the argmax
            output = output.argmax(dim=-1).cpu().numpy().tolist()
            # Convert the target output sequence to a list for comparison
            tgt_output = tgt_output.cpu().numpy().tolist()

            # Compare each predicted sequence with its corresponding target sequence
            for pred, target in zip(output, tgt_output):
                # Extract the predicted sequence up to the first <EOS> token
                pred_seq = []
                for tok in pred:
                    if tok == tgt_eos_idx:
                        break
                    pred_seq.append(tok)
                
                # Extract the target sequence up to the first <EOS> token
                target_seq = []
                for tok in target:
                    if tok == tgt_eos_idx:
                        break
                    target_seq.append(tok)
                
                # Check if the entire predicted sequence matches the target sequence
                if pred_seq == target_seq:
                    correct_sequences += 1  # Increment correct sequence counter
                
                total_sequences += 1  # Increment total sequence counter

    # Calculate and return sequence-level accuracy
    return correct_sequences / total_sequences

# Updated to Exp3 only for JUMP!!
# provides a structured way to retrieve paths to datasets of varying sizes
def get_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3 (JUMP only)."""
    base_path = "data/add_prim_split/with_additional_examples"
    sizes = ["0", "1", "2", "4", "8", "16", "32"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_addprim_complex_jump_num{size}_rep1.txt"
        test_path = "data/add_prim_split/tasks_test_addprim_jump.txt"
        pairs.append((train_path, test_path, size))
    return pairs


# Updated for Exp3
# trains and evaluates the Transformer model on a specific training and testing dataset.
def main(train_path, test_path, model_suffix):
    """Modified main function accepting dataset paths"""
    # Hyperparameters Experiment 3
    EMB_DIM = 128
    N_LAYERS = 2         # Updated to 2 layers
    N_HEADS = 8
    FORWARD_DIM = 256    # Updated to 256
    DROPOUT = 0.15       # Increased dropout
    LEARNING_RATE = 2e-4 # Updated learning rate
    BATCH_SIZE = 16      # Reduced batch size
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset with provided paths
    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16)
    
    # Model Initialization
    model = Transformer(
        src_vocab_size=dataset.src_vocab.vocab_size,
        tgt_vocab_size=dataset.tgt_vocab.vocab_size,
        src_pad_idx=dataset.src_vocab.tok2id["<PAD>"],
        tgt_pad_idx=dataset.tgt_vocab.tok2id["<PAD>"],
        emb_dim=EMB_DIM,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        forward_dim=FORWARD_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Cross-entropy loss ignoring PAD
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab.tok2id["<PAD>"])

    # optimizer uses AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_token_accuracy = 0.0
    best_sequence_accuracy = 0.0
    # Training and Evaluation
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate(model, test_loader, criterion, DEVICE)
        token_accuracy = calculate_accuracy(model, test_loader, dataset, DEVICE)
        sequence_accuracy = calculate_sequence_accuracy(model, test_loader, dataset, DEVICE)
        
        print(f"Dataset p{model_suffix} - Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Token-Level Accuracy: {token_accuracy:.4f}")
        print(f"Sequence-Level Accuracy: {sequence_accuracy:.4f}")

        
        # Best Token-Level Accuracy
        if token_accuracy > best_token_accuracy:
            best_token_accuracy = token_accuracy
            print(f"Best token accuracy: {best_token_accuracy:.4f}")

        # Best Sequence-Level Accuracy
        if sequence_accuracy > best_sequence_accuracy:
            best_sequence_accuracy = sequence_accuracy
            print(f"Best sequence accuracy: {best_sequence_accuracy:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'token_accuracy': token_accuracy,
                'sequence_accuracy': sequence_accuracy,
            }, f'best_model_p{model_suffix}.pt')
                    
        print("-" * 50)

    print(f"Training completed for p{model_suffix}. Best token accuracy: {best_token_accuracy:.4f}, Best sequence accuracy: {best_sequence_accuracy:.4f}")
    return best_token_accuracy, best_sequence_accuracy


# Updated for Exp3
# This function automates training and evaluation across datasets of varying sizes.
def run_all_variations():
    """Run training for all dataset size variations and plot results."""
    token_results = []
    sequence_results = []
    sizes = []

    for train_path, test_path, size in get_dataset_pairs():
        print(f"\nStarting training for dataset size p{size}")
        print("=" * 70)
        token_accuracy, sequence_accuracy = main(train_path, test_path, size)
        token_results.append(token_accuracy)
        sequence_results.append(sequence_accuracy)
        sizes.append(size)
    
    # Plot Token-Level Accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.bar(sizes, [t * 100 for t in token_results])
    plt.xlabel("Number of Composed Commands Used For Training")
    plt.ylabel("Token-Level Accuracy (%)")
    plt.title("Token-Level Accuracy")
    plt.show()
    
    # Plot Sequence-Level Accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(sizes, [s * 100 for s in sequence_results])
    plt.xlabel("Number of Composed Commands Used For Training")
    plt.ylabel("Sequence-Level Accuracy (%)")
    plt.title("Sequence-Level Accuracy")
    plt.show()

    # Print Results Summary
    print("\nFinal Results Summary:")
    print("=" * 30)
    for size, token_acc, seq_acc in zip(sizes, token_results, sequence_results):
        print(f"Dataset {size}: Token Accuracy: {token_acc:.4f}, Sequence Accuracy: {seq_acc:.4f}")


if __name__ == "__main__":
    run_all_variations()


    # Training time per epoch (experiment 3, M3 Pro Chip with MPS GPU): 7min-11min
    # Inference time (experiment 3, turn_left, M3 Pro Chip with MPS GPU): 2min
    # Inference time (experiment 3, jump, per composition, M3 Pro Chip with MPS GPU): 28min

