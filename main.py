# main.py
import argparse
import pandas as pd
import os

from utils import set_seed, run_preprocessing
from train import run_train
from test import run_test
from args import CFG

def main(args):
    # Update configuration based on arguments.
    CFG.seed = 42
    set_seed(CFG.seed)
    if args.epochs:
        CFG.epochs = args.epochs
    if args.batch_size:
        CFG.batch_size = args.batch_size
    if args.model_name:
        CFG.model_name = args.model_name
    if args.preprocessing:
        CFG.preprocessing = args.preprocessing
    if args.load_data is not None:
        # If preprocessing method is on_the_fly, force LOAD_DATA False.
        if args.preprocessing == "on_the_fly":
            CFG.LOAD_DATA = False
        else:
            CFG.LOAD_DATA = args.load_data

    CFG.datasets=args.datasets
    if args.datasets == 'origin':
        # Load the training CSV.
        train_df = pd.read_csv(CFG.train_csv)
    elif args.datasets == 'modify':
        train_csv = os.path.expanduser('~/Dataset/CV/birdclef-2025/train_final_df.csv')
        train_df = pd.read_csv(train_csv)

    if args.mode == "pre":
        # Run pre-processing: generate and save spectrograms.
        run_preprocessing(train_df, CFG, CFG.spectrogram_npy)
        
    elif args.mode == "train":
        run_train(train_df, CFG, args)
        
    elif args.mode == "test":
        if not args.checkpoint:
            print("Checkpoint path is required for test mode.")
            return
        run_test(CFG, args.checkpoint, train_df, args)  # For demo, using train_csv as test.
        
    elif args.mode == "train_test":
        run_train(train_df, CFG, args)
        if not args.checkpoint:
            print("Checkpoint path is required for test mode after training.")
            return
        run_test(CFG, args.checkpoint, train_df, args)
    else:
        print("Unrecognized mode.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main entry: choose mode among 'pre', 'train', 'test', 'train_test'"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["pre", "train", "test", "train_test"],
        help="Operation mode: pre (preprocessing), train, test, or train_test"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint for test"
        )
    # Additional arguments can override configuration options.
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["origin", "modify"],
        default='origin'
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Batch size"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        help="Model name from timm"
    )
    parser.add_argument(
        "--preprocessing", 
        type=str, 
        choices=["precomputed", "on_the_fly"],
        help="Preprocessing method"
    )
    parser.add_argument(
        "--no_load_data", 
        action="store_false", 
        dest="load_data",
        help="If set, generate spectrograms on-the-fly"
    )
    args = parser.parse_args()
    
    main(args)
