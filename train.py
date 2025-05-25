import os
import torch
from torch.utils.data import DataLoader
from dataset import TokenDataset
from optimizer import AdamW
from loss import cross_entropy_loss
from model import TransformerModel

tinystories_config = {
    "vocab_size": 50257,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "num_layers": 4,
    "num_heads": 16,
    "lr": 1e-3,
    "batch_size": 32,
    "train_dataset": 'data/tinystories/train-dataset.npy',
    "val_dataset": 'data/tinystories/val-dataset.npy',
}

def createModel(config, device):
    return TransformerModel(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        device=device,
        )

class Trainer:
    def __init__(self, session_name, config, model, device, max_steps=-1, eval_interval=1000, checkpoint=None):
        self.train_dataset = TokenDataset(config['train_dataset'], config['context_length'], device=device)
        self.val_dataset = TokenDataset(config['val_dataset'], config['context_length'], device=device)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False)

        self.session_name = session_name
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=config['lr'])
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.loss_values = []
        self.min_eval_loss = float('inf')
        self.step = 1


    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.step,
            'eval_loss': self.min_eval_loss,
            'loss_values': self.loss_values,
        }
        os.makedirs(os.path.dirname(self.session_name + '/'), exist_ok=True)
        torch.save(checkpoint, f"{self.session_name}/{self.step}.pth")


    def load_checkpoint(self, src):
        checkpoint = torch.load(src, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['iteration']
        self.min_eval_loss = checkpoint['eval_loss']
        self.loss_values = checkpoint['loss_values']


    def evaluate(self, max_steps=-1):
        eval_loss = 0
        n_iter = 1
        self.model.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                eval_loss += cross_entropy_loss(self.model(inputs), targets)
                n_iter += 1
                if max_steps > 0 and n_iter >= max_steps:
                    break

        self.model.train()
        return eval_loss / n_iter


    def train(self):
        self.model.train()
        acc_loss = 0

        for batch in self.train_loader:
            inputs, targets = batch
            loss = cross_entropy_loss(self.model(inputs), targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_loss += loss.item()
            self.loss_values.append(loss.item())

            if self.step % self.eval_interval == 0:
                eval_loss = self.evaluate(max_steps=1000)
                print(f"step {self.step}: {acc_loss/self.eval_interval:.4f} / {eval_loss:.4f}")
                acc_loss = 0
                if (eval_loss < self.min_eval_loss):
                    self.min_eval_loss = eval_loss
                    self.save_checkpoint()

            if self.max_steps > 0 and self.step >= self.max_steps:
                break
            self.step += 1

def main():
    config = tinystories_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    model = torch.compile(createModel(config, device))
    trainer = Trainer("lr=1e-3_shuffle", config, model, device, max_steps=12000, eval_interval=2000)
    trainer.train()


if __name__ == "__main__":
    main()