import argparse
from time import time
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from datetime import datetime
from utils import * 
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def test(args):
    model = SentenceTransformer(args.model_name)
    predictions, test_labels= [] ,[]    #Truncate long passages to 512 tokens
    top_k = args.top_k                          #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    print('Start reading the queries')
    contexts, questions, qc_map = read_data(args.path_test)
    corpus_embeddings = model.encode(list(contexts.values()), convert_to_tensor=True, show_progress_bar=True)
    print("Finished embedding the corpus")
    
    if args.cross_encoder:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    t0 = time()
    with tqdm(questions.items()) as tq:
        
        for qid , query in tq:
            
            query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)[1].cpu().detach().numpy()
            if args.cross_encoder:
                    cross_inp = [[query, contexts[idx]] for idx in top_results]
                    cross_scores = cross_encoder.predict(cross_inp)
                    top_results =top_results[np.flip(np.argsort(cross_scores))[:top_k]]

            predictions.append(top_results)
            test_labels.append(qc_map[qid])
        
            acc,  mrr, _ = compute_errors(test_labels, predictions)
            tq.set_postfix(accuracy = acc,mrr = mrr)
    t = time() - t0
    print(f"The Accuracy of our model is : {acc}",
              f"\n The MRR@{top_k} of our model is : {mrr}")
    print('TIME ELAPSED ', t)
    print(f'\n Speed : {len(questions)/t}queries/sec')
    acc,  mrr, list_accuracies = compute_errors(test_labels, predictions)
    plt.figure()
    plt.plot(list(range(1, len(predictions[0])+1)), list_accuracies)
    plt.title('Accuracy with respect to the number of retrieved documents')
    plt.savefig("Accuracy_evolution.jpg")
    save_json(args.path_result, 'test', {"labels": test_labels, "predictions": list(predictions), "time": t, "accuracy": acc, "MRR": mrr, "list_accuracies": list_accuracies, "speed": len(questions)/t})
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_test", type=str, default="/content/data/squad/dev-v1.1.json", 
        help="data folder name") 
    parser.add_argument("-r", "--path_result", type=str, default="results", 
        help="data folder name")
    parser.add_argument("-m","--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", 
        help="Model to encode text")
    parser.add_argument("-t", "--top_k", type=int, default=10, 
        help="Top documents to return")
    parser.add_argument("-c", "--cross_encoder", type=bool, default=False, 
        help="Whether to use cross encoder to refine the result or not")
    test(parser.parse_args())
    