from trainer import Trainer
import argparse


# options
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--validation", default=False, action='store_true')
parser.add_argument("--pretrain_model", default=None)
args = parser.parse_args()
# initialize the trainer
Train = Trainer(batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                split=args.split)
if args.validation:
    # test the model
    Train.sample(args.pretrain_model)
else:
    # train the model
    Train.train()