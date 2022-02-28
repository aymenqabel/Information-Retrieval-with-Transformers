# Information-Retrieval-with-Transformers
Information Retrieval (IR)is finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers).

During this challenge, I created an easy-to-train Information retrieval system based on Sentence-Transformers library.

## Setup
``` bash
# Clone this repository
git clone https://github.com/ayoumen/Information-Retrieval-with-Transformers.git
cd Information-Retrieval-with-Transformers/
# Install packages
pip install -r requirements.txt
```

## Data
The data that we used is SQuAD v1.1, a dataset that consists of 100,000+ question and answer pairs on 500+ articles.

## How to use it?
If you have an annotated dataset, you can train the model to increase the performance on the specialized task:
``` bash
python train.py -d path_to_trainset 
```
You can specify the parameter of the training process, such us the number of epochs and the validation set.

If you want to train/make a prediction using a model form HuggingFace repository, you can specify the name of the model in the parameter --model_name
