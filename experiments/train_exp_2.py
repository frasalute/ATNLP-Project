import sys
import json
import matplotlib.pyplot as plt
sys.path.append('/work/ATNLP-Project')
from testing import train
import torch
import numpy as np
from dataset import SCANDataset


def save_results(results, filename="experiment2_results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def plot_histogram(accuracy, filename="experiment2_histogram.png"):
    """Plot histogram for sequence accuracy."""
    plt.figure(figsize=(10, 6))
    plt.bar(["Sequence Accuracy"], [accuracy], color="orange", alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("Sequence Accuracy for Experiment 2")
    plt.savefig(filename)
    print(f"Histogram saved to {filename}")
    plt.close()


def run_experiment():
    """Run training for Experiment 2."""
    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    model_suffix = "length"

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Run training
    print(f"Starting training for Experiment 2 (Length Split)")
    print("=" * 50)

    try:
        model, token_acc, seq_acc = train(
            train_path=train_path,
            test_path=test_path,
            hyperparams=hyperparams,
            model_suffix=model_suffix,
            random_seed=seed,
        )

        # Save results
        results = {
            "token_accuracy": token_acc,
            "sequence_accuracy": seq_acc,
        }
        save_results(results)

        # Plot histogram
        plot_histogram(seq_acc)

    except Exception as e:
        print(f"Error during Experiment 2: {e}")


if __name__ == "__main__":
    run_experiment()
