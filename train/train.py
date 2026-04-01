import os
from dotenv import load_dotenv
import wandb
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
)
from config import (
    OUTPUT_DIR, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, COMPUTE_DTYPE,
    EARLY_STOPPING_PATIENCE, DATALOADER_NUM_WORKERS, MODEL_SAVE_PATH
)
from data_utils import load_and_preprocess_data, get_tokenizer
from model_utils import (
    get_quantization_config, load_model_and_apply_lora, 
    get_data_collator, compute_metrics
)

load_dotenv()
PROJECT_NAME = os.getenv("PROJECT_NAME")
ENTITY_NAME = os.getenv("ENTITY_NAME")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def main():
    run = wandb.init(
        entity=ENTITY_NAME,
        project=PROJECT_NAME,
        config={
            "dataset": "cnn_dailymail",
            "model": "google/flan-t5-base",
            "epochs": NUM_TRAIN_EPOCHS
        },
    )

    tokenizer = get_tokenizer()
    train_dataset, eval_dataset = load_and_preprocess_data(tokenizer)

    quant_config = get_quantization_config()
    model = load_model_and_apply_lora(tokenizer, quant_config)
    data_collator = get_data_collator(tokenizer, model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        
        logging_steps=EVAL_STEPS,
        report_to="wandb",
        predict_with_generate=True,
        generation_max_length=128,

        bf16=(COMPUTE_DTYPE == "bfloat16"), 
        fp16=(COMPUTE_DTYPE == "float16"),
        
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        group_by_length=True,
        optim="paged_adamw_32bit",
        include_inputs_for_metrics=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    trainer.train()

    trainer.save_model(MODEL_SAVE_PATH) 
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    run.finish()

if __name__ == "__main__":
    main()