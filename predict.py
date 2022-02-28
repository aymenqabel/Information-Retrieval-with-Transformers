import argparse
from threading import main_thread
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from utils import * 
import torch
import os.path
from os import path
from time import time
import numpy as np

def predict(args):
    download_data()
    model = SentenceTransformer(args.model_name)
    top_k = args.top_k                          #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    print('Start reading the queries')
    contexts_train, _, _ = read_data(args.path_test)
    contexts_test, _, _ = read_data(args.path_train)
    all_contexts = list(contexts_train.values()) + list(contexts_test.values())
    # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
    if not(path.exists("corpus_embeddings.pt")):
        print("Indexing the documents")
        corpus_embeddings = model.encode(all_contexts, convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, 'corpus_embeddings.pt')
        print("Finished embedding the corpus")
    else:
        corpus_embeddings = torch.load("corpus_embeddings.pt")
    
    
    if args.cross_encoder:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    while 1:
      top_k_show = 10
      print("Enter your query: , top docs to show")
      query = input()
      t0 = time()
      if query == 'exit':
        break
      test_query = query.split(',')
      if len(test_query)>1:
        try:
          m = len(test_query[-1])
          top_k_show = int(test_query[-1])
          if top_k_show > top_k:
            print(f"Top docs to show must be smaller than {top_k}. Set to default: 10")
            top_k_show = 10
          query = query[:-m-1]
        except:
          pass
      

      print(query)
      query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True)
      cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
      top_results = torch.topk(cos_scores, k=top_k)
      cross_scores = top_results[0].cpu().detach().numpy()
      sorted_cross = list(range(top_k_show))
      top_results = top_results[1].cpu().detach().numpy()
      if args.cross_encoder:
          print("Using Cross Encoder")
          cross_inp = [[query, all_contexts[idx]] for idx in top_results]
          cross_scores = cross_encoder.predict(cross_inp)
          sorted_cross = np.flip(np.argsort(cross_scores))[:top_k_show]
          top_results =top_results[sorted_cross]
      print("\n-------------------------\n")
      print(f"Top-{top_k} contexts")
      for i in range(top_k_show):
          print("\t{:.3f}\t{}".format(cross_scores[sorted_cross[i]], all_contexts[top_results[i]].replace("\n", " ")))
      print(f"Documents retrieved in {time()-t0} seconds")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--path_train", type=str, default="/content/data/squad/train-v2.0.json", 
        help="data folder name") 
    parser.add_argument("-p", "--path_test", type=str, default="/content/data/squad/dev-v2.0.json", 
        help="data folder name") 
    parser.add_argument("-m","--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", 
        help="Model to encode text")
    parser.add_argument("-t", "--top_k", type=int, default=10, 
        help="Top documents to return")
    parser.add_argument("-c", "--cross_encoder", type=bool, 
        help="Whether to use cross encoder to refine the result or not")
    parser.add_argument("-q", "--query", type=str, default='Who is Beyoncé', 
        help="Question to look for")

    predict(parser.parse_args())
    