import torch
import numpy as np
import nltk
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM, BitsAndBytesConfig, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import (
    MODEL_NAME, COMPUTE_DTYPE, 
    BNB_4BIT_QUANT_TYPE, LORA_R, LORA_ALPHA, LORA_DROPOUT, 
    TARGET_MODULES, MAX_INPUT_LENGTH
)

def check_nltk_resource(resource_name='punkt'):
    try:
        nltk.data.find(f'tokenizers/{resource_name}')
    except LookupError:
        nltk.download(resource_name, quiet=True)

def get_quantization_config():
    ''' настройка квантования '''
    
    compute_dtype = getattr(torch, COMPUTE_DTYPE)
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype
    )

def load_model_and_apply_lora(tokenizer, quantization_config):
    ''' применение lora '''

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters() # количество обучаемых параметров
    
    return peft_model

def get_data_collator(tokenizer, model):
    ''' сборка батчей '''
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest", 
        max_length=None
    )

def compute_metrics(eval_preds, tokenizer):
    ''' оценка качества после каждой эпохи '''
    check_nltk_resource('punkt')

    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_metric = evaluate.load("rouge")
    result = rouge_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    
    return {k: round(v, 4) for k, v in result.items()}