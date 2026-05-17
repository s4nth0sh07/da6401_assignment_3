"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from model import Transformer
from lr_scheduler import NoamScheduler



# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """
    
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction = "sum")
        temp = 1.0
        self.confidence = temp-smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        true_distance = torch.zeros_like(logits)
        temp = self.smoothing
        temp = temp / (self.vocab_size - 1)
        true_distance.fill_(temp)
        temp = target.unsqueeze(1)
        args = (1, temp, self.confidence)
        true_distance.scatter_(*args)
        idx = (slice(None), self.pad_idx)
        true_distance[idx]= 0.0
        check = (target == self.pad_idx)
        mask = torch.nonzero(check)
        if mask.dim() > 0:
            temp = mask.squeeze()
            args = (0, temp, 0.0)
            true_distance.index_fill_(*args)
        
        temp = F.log_softmax(logits, dim = -1)
        args = (temp, true_distance)
        loss = self.criterion(*args)
        return loss



# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()

    stats = {'loss' : 0.0, 'tokens' : 0}
    for batch in data_iter:
        src, tgt = (batch[k].to(device) for k in ('src', 'tgt'))
        tgt_in, tgt_y= tgt[:, :-1], tgt[:, 1:]
        masks = {
            'src_mask': make_src_mask(src, pad_idx = 1).to(device),
            'tgt_mask': make_tgt_mask(tgt_in, pad_idx = 1).to(device)
        }
        logits= model(src, tgt_in, **masks)
        temp  = logits.reshape(-1, logits.size(-1))
        args = (temp, tgt_y.reshape(-1))
        loss = loss_fn(*args)

        if is_train:
            for step in (optimizer.zero_grad, loss.backward,optimizer.step):
                step()
            scheduler and scheduler.step()
        
        stats['loss'] += loss.item()
        check = (tgt_y != 1)
        stats['tokens'] += (check).sum().item()
    res = stats['loss'] / stats['tokens']
    return res



# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    memory = model.encode(src, src_mask)
    temp = torch.ones(1, 1)
    temp = temp.fill_(start_symbol)
    temp = temp.type(torch.long)
    ys = temp.to(device)
    for i in range(max_len -1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        args = (memory, src_mask, ys, tgt_mask)
        out = model.decode(*args)
        idx = (slice(None), -1, slice(None))
        prob = out[idx]
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        new_token = torch.ones(1, 1)
        new_token = new_token.type_as(src.data)
        new_token = new_token.fill_(next_word)
        tensors_to_cat = [ys, new_token]
        ys = torch.cat(tensors_to_cat, dim = 1)
        check = (next_word == end_symbol)
        if check == True:
            break
    
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()
    candidate_corpus = []
    references_corpus = []
    
    SOS_IDX = 2
    EOS_IDX = 3
    
    # Get the reverse lookup dictionary (Integers -> Strings)
    itos = tgt_vocab.get_itos()

    print("Evaluating BLEU Score on Test Set...")
    with torch.no_grad():
        for batch in test_dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Create the source mask
            # Make sure make_src_mask is imported from model.py!
            src_mask = make_src_mask(src, pad_idx=1).to(device)

            for i in range(src.size(0)):
                # Extract the single source sentence and add a dummy batch dimension [1, seq_len]
                single_src = src[i].unsqueeze(0)
                single_src_mask = src_mask[i].unsqueeze(0)
                
                # 1. Generate the prediction using greedy_decode
                pred_indices = greedy_decode(
                    model, 
                    single_src, 
                    single_src_mask, 
                    max_len, 
                    SOS_IDX, 
                    EOS_IDX, 
                    device
                )
                
                # 2. Squeeze the prediction and target to 1D lists
                # Drop the first token (<sos>) for both
                pred_list = pred_indices.squeeze(0).tolist()[1:]
                target_list = tgt[i].tolist()[1:]
                
                # 3. Convert integers to string words
                pred_words = []
                for idx in pred_list:
                    if idx == EOS_IDX:
                         break
                    # The Solution: Use itos.get() and default to '<unk>' if not found
                    pred_words.append(itos.get(idx, '<unk>')) 

                target_words = []
                for idx in target_list:
                    if idx == EOS_IDX:
                        break
                    # The Solution: Filter out padding, then lookup the string
                    if idx != 1:
                        target_words.append(itos.get(idx, '<unk>'))

                candidate_corpus.append(" ".join(pred_words))
                references_corpus.append(" ".join(target_words))

    import bleu
    final_bleu = bleu.list_bleu([references_corpus], candidate_corpus)
    
    return float(final_bleu)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)

    print(f"Saving checkpoint to {path} at epoch {epoch}")

    src_vocab_size = model.src_embed.num_embeddings
    tgt_vocab_size = model.tgt_embed.num_embeddings

    N = len(model.encoder.layers)

    model_config = {
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': model.d_model,
        'N': N,
        'num_heads': 8,   
        'd_ff': 2048,
        'dropout': 0.1
    }

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config,
        'src_vocab': dict(model.src_vocab),
        'tgt_vocab': dict(model.tgt_vocab),
        'tgt_itos':  {v: k for k, v in model.tgt_vocab.items()}
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    print(f"Loading checkpoints from {path}")
    args = (path, )
    kwargs = {"map_location" : 'cpu'}
    checkpoint = torch.load(*args, **kwargs)
    temp = checkpoint['model_state_dict']
    model.load_state_dict(temp)

    if optimizer is not None:
        'optimizer_state_dict' in checkpoint and optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint:
            checkpoint['scheduler_state_dict'] is not None and scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    res = checkpoint.get('epoch', 0)
    return res


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    config = {
        'batch_size': 128, 
        'num_epochs': 120,
        'd_model': 256,
        'N': 3,
        'num_heads': 8,
        'd_ff': 512,
        'dropout': 0.2,
        'smoothing': 0.1,
        'warmup_steps': 4000
    }
    
    wandb.init(project="da6401-a3", config=config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("Initializing datasets...")
    train_data = Multi30kDataset(split='train')
    val_data = Multi30kDataset(split='validation', src_vocab=train_data.src_vocab, tgt_vocab=train_data.tgt_vocab)
    test_data = Multi30kDataset(split='test', src_vocab=train_data.src_vocab, tgt_vocab=train_data.tgt_vocab)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    model.src_vocab = train_data.src_vocab
    model.tgt_vocab = train_data.tgt_vocab
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    scheduler = NoamScheduler(optimizer, config['d_model'], config['warmup_steps'])
    
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab), 
        pad_idx=1, 
        smoothing=config['smoothing']
    ).to(device)

    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        print(f"Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss, 
            "val_loss": val_loss
        })

        # 1. Save the best validation loss checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"⭐ New best validation loss ({best_val_loss:.4f})! Saving best_model.pt...")
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_model.pt")

        # 2. Save the final epoch checkpoint
        if epoch == config['num_epochs']:
            print(f"🏁 Final epoch reached! Saving last_model.pt...")
            save_checkpoint(model, optimizer, scheduler, epoch, path="last_model.pt")

    
    print("\nLoading best_model.pt for final BLEU evaluation...")
    load_checkpoint("best_model.pt", model)
    bleu = evaluate_bleu(model, test_loader, train_data.tgt_vocab, device=device)
    print(f"Final Test BLEU Score: {bleu:.2f}")
    
    wandb.log({'test_bleu': bleu})
    
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
