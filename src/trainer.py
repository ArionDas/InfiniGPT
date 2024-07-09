import os
import matplotlib.pyplot as plt

import torch
import urllib.request
import tiktoken

from infini_gpt_config import INFINIGPT_CONFIG
from infini_transformer import InfiniTransformer
from infinigpt import InfiniGPT
from dataloader import InfiniGPT_dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) ## suitable for multi-class & multi-label classifications
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    step = -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            
            optimizer.zero_grad()
            
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            loss.backward() ## calculating loss gradients
            
            optimizer.step() ## updating model weights using loss gradients
            
            tokens_seen += input_batch.numel() ## .numel() gets the number of elements in a PyTorch tensor. Returns an integer regardless of dimension
            
            step += 1
            
            
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()       
            
            
def main(config):
    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """ Text Data """
    
    url = "https://github.com/ArionDas/InfiniGPT/blob/eb3abdc6eaf1d8f17be7e92e81fd641a710aae26/data/book_clean.txt"
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    
    
    """ Model """
        
    model = InfiniGPT(INFINIGPT_CONFIG)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"]
    ) ## weight decay somewhat helps in preventing overfitting
    
    """ Dataloaders """
    train_ratio = 0.90
    split = int(train_ratio * len(text_data))
    
    train_loader = InfiniGPT_dataloader(
        text_data[:split],
        batch_size = config["batch_size"],
        max_length = config["context_length"],
        stride = config["stride"],
        drop_last = True,
        shuffle = True,
    )
    
    val_loader = InfiniGPT_dataloader(
        text_data[split:],
        batch_size = config["batch_size"],
        max_length = config["context_length"],
        stride = config["stride"],
        drop_last = False,
        shuffle = True,
    )
    
    """ Model Training """
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_losses, val_losses, tokens_seen = train_model(model, train_loader, val_loader, optimizer, device,
                                                       num_epochs=config["num_epochs"], eval_freq=1, eval_iter=1,
                                                       tokenizer=tokenizer)
    
    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    
    config = INFINIGPT_CONFIG
    
    train_losses, val_losses, tokens_seen, model = main(config)
    
    epochs_tensor = torch.linspace(0, config["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.show() 
    
    torch.save(model.state_dict(), "model/sample_model.pth")         
    