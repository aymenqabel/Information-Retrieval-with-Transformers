import argparse
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from datetime import datetime
from dataset import SQuAD


from utils import *


def train(args):
    
    # Load our embedding model
    print("use pretrained SBERT model")
    model = SentenceTransformer(args.name_model)
    model.max_seq_length = args.seq_length

    model_save_path = 'output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    corpus, questions, qcmap = read_data(args.path_train)

    train_dataset = SQuAD(questions, corpus, qcmap)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            use_amp=True,
            checkpoint_path=model_save_path,
            checkpoint_save_steps=len(train_dataloader),
            optimizer_params = {'lr': args.lr},
            )

    # Save the model
    model.save(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="ehr", 
        help="data folder name")
    parser.add_argument("-m","--model_name", type=str, default="distilbert-base-uncased", 
        help="Model to fine tune")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("-b", "--batch_size", type=int, default=64, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-s", "--seq_length", type=int, default=512, 
        help="Maximum length of sentences")
    parser.add_argument("--warmup_steps", default=1000, type=int)

    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="prints training loss every k batch")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="model learning rate")
    train(parser.parse_args())