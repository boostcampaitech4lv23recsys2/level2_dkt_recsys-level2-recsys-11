import os
import pytorch_lightning as pl
import torch
from args import parse_args
from src import trainer
from src.dataloader import Preprocess, get_loaders
from dkt_lightning import DktLightning
import sys
sys.path.append('./')

def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    _, test_loader = get_loaders(args, None, test_data)
    
    model = DktLightning.load_from_checkpoint(
        os.path.join('output/epoch=19-val_loss=0.00.ckpt'), args=args
    )
    
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=test_loader)
    print(predictions)
    # model = trainer.load_model(args).to(args.device)
    # trainer.inference(args, test_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
