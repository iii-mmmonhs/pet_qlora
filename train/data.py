import nltk
from datasets import load_dataset
from transformers import AutoTokenizer
from config import (
    DATASET_NAME, DATASET_CONFIG, MAX_INPUT_LENGTH, 
    MAX_TARGET_LENGTH, MODEL_NAME, TRAIN_SIZE, EVAL_SIZE
)

def load_and_preprocess_data(tokenizer, TRAIN_SIZE=4500, EVAL_SIZE=4500):
    ''' загрузка и предобработка датасета '''
        
    def preprocess_function(examples):
        ''' промпт и токенизация '''
        inputs = [f"summarize news article: {doc}" for doc in examples["article"]]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True, 
            padding=False
        )
        
        labels = tokenizer(
            text_target=examples["highlights"], 
            max_length=MAX_TARGET_LENGTH, 
            truncation=True, 
            padding=False
        )
        
        return model_inputs
    
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    train_dataset = dataset["train"].select(range(TRAIN_SIZE))
    eval_dataset = dataset["test"].select(range(EVAL_SIZE))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=64,
        num_proc=4,
        remove_columns=["article", "highlights"],
        load_from_cache_file=False
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=64,
        num_proc=4,
        remove_columns=["article", "highlights"],
        load_from_cache_file=False
    )

    return train_dataset, eval_dataset

def get_tokenizer():
    ''' загрузка токенизатора '''
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    return tokenizer