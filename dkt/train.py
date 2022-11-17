import os

from dkt_lightning import DktLightning
import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds

# import lightning as pl
import pytorch_lightning as pl


def main(args):
    from args import parse_args
    from src.dataloader import Preprocess, get_loaders
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb.login()

    # 01. Make Dataloader
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # 02. Define model
    # wandb.init(project="dkt", config=vars(args))
    model = DktLightning(args)
    
    # 03. Define Trainer & Callbacks
    wandb_logger = pl.loggers.WandbLogger(project='dkt')
    model_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor='validate_acc',
        filename='{epoch}-{val_loss:.4f}',
        mode='max',
        save_top_k=3
    )
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        logger=wandb_logger,
        max_epochs=args.n_epochs,
        gradient_clip_val=args.clip_grad,
        callbacks=[
            model_ckpt,
        ]
    )
    
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":

    
    args = parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
