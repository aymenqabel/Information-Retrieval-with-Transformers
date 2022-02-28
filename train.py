import argparse
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,losses
from datetime import datetime
from dataset import SQuAD
from utils import *
from evaluator import accuracyevaluator
import matplotlib.pyplot as plt

def train(args):
    download_data()
    # Load our embedding model
    print("Load the Model ", args.model_name)
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.seq_length
    print("Model loaded successfully")
    model_save_path = 'output/train_encoder-{}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Reading the data")
    corpus, questions, qcmap = read_data(args.path_train)
    print("Data read succesfully")
    train_dataset = SQuAD(questions, corpus, qcmap)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    print('Fitting the model ')
    contexts_test, questions_test, qc_map_test = read_data(args.path_test)
    evaluator = accuracyevaluator(contexts_test, questions_test, qc_map_test)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator = evaluator, 
            epochs=args.epochs,
            steps_per_epoch = args.steps_per_epoch, 
            warmup_steps=args.warmup_steps,
            use_amp=True,
            evaluation_steps = args.evaluation_steps,
            checkpoint_path=model_save_path,
            checkpoint_save_steps=len(train_dataloader),
            optimizer_params = {'lr': args.learning_rate},
            )
    print("Model fitted and ready to use")
    plt.figure()
    plt.plot(evaluator.accuracy)
    plt.plot(evaluator.mrrlist)
    plt.savefig("output/evaluation.jpg")
    # Save the model
    model.save(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_train", type=str, default="/content/data/squad/train-v1.1.json", 
        help="data folder name")
    parser.add_argument("-m","--model_name", type=str, default="distilbert-base-uncased", 
        help="Model to fine tune")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("-b", "--batch_size", type=int, default=16, 
        help="dataset batch size")
    parser.add_argument("-e", "--epochs", type=int, default=2, 
        help="number of epochs")
    parser.add_argument("-x", "--steps_per_epoch", type=int, default=20, 
        help="number of steps per epoch")
    parser.add_argument("-v", "--evaluation_steps", type=int, default=50, 
        help="number of steps before evaluation")
    parser.add_argument("-s", "--seq_length", type=int, default=512, 
        help="Maximum length of sentences")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("-p", "--path_test", type=str, default="/content/data/squad/dev-v1.1.json", 
        help="path to dev data")
    parser.add_argument("-k", "--print_every_k_batch", type=int, default=1, 
        help="prints training loss every k batch")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, 
        help="model learning rate")
    
    
    train(parser.parse_args())