import logging
import random
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset
import evaluate
from config import (
    MODEL_NAME, LORA_ADAPTERS_PATH, DATASET_NAME, DATASET_CONFIG, 
    MAX_NEW_TOKENS, MAX_INPUT_LENGTH, MAX_TOKENS_JUDGE, INFERENCE_DEVICE
)
from .judge import Judge

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Управление экспериментом сравнения моделей.
    
    Отвечает за загрузку моделей, подготовку данных, 
    генерацию предсказаний и расчет метрик.
    """
    
    def __init__(self):
        self.tokenizer = None
        self.peft_model = None
        self.dataset = None
        self.rouge = None
        self._is_loaded = False
    
    def load_resources(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
        device_map = "auto" if INFERENCE_DEVICE == "cuda" else None
            
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map
        )
            
        if os.path.exists(LORA_ADAPTERS_PATH):
            self.peft_model = PeftModel.from_pretrained(base_model, LORA_ADAPTERS_PATH)
        else:
            logger.warning(f"LoRA адаптеры не найдены в {LORA_ADAPTERS_PATH}")
            self.peft_model = base_model
            
        self.peft_model.eval()
            
        self.dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=True)
        self.rouge = evaluate.load("rouge")
            
        self._is_loaded = True
    
    def run(self):
        """
        Выполняет полный цикл эксперимента: выбирает случайный пример, 
        генерирует суммаризации, подсчитывает метрики, получает оценку судьей.
        """
        try:
            seed = random.randint(0, 10000)
            example = next(iter(self.dataset.shuffle(seed=seed)))
            
            source_text = example['article']
            reference_text = example['highlights']
            
            inputs = self.tokenizer(
                f'summarize: {source_text}', 
                return_tensors="pt", 
                max_length=MAX_INPUT_LENGTH, 
                truncation=True
            ).to(self.peft_model.device)
            
            logger.info("Генерация резюме qlorа")
            with torch.no_grad():
                out_q = self.peft_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            q_sum = self.tokenizer.decode(out_q[0], skip_special_tokens=True)
            logger.info(f"Model A результат: {q_sum[:50]}")
            
            logger.info("Генерация резюме базовой модели")
            with torch.no_grad():
                with self.peft_model.disable_adapter():
                    out_b = self.peft_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            b_sum = self.tokenizer.decode(out_b[0], skip_special_tokens=True)
            logger.info(f"Model B результат: {b_sum[:50]}")
            
            q_metrics = self.rouge.compute(predictions=[q_sum], references=[reference_text])
            q_metrics = {k: round(v, 4) for k, v in q_metrics.items()}
            
            b_metrics = self.rouge.compute(predictions=[b_sum], references=[reference_text])
            b_metrics = {k: round(v, 4) for k, v in b_metrics.items()}
            
            logger.info("Запрос к судье")
            judge_results = Judge.evaluate(source_text, reference_text, q_sum, b_sum)
            
            summary_msg = f"Победитель: {judge_results.get('winner')}. {judge_results.get('reason')}"
            
            return (
                source_text, reference_text,
                q_sum, q_metrics,
                b_sum, b_metrics,
                judge_results,
                summary_msg
            )
            
        except Exception as e:
            logger.error(f"Ошибка при запуске эксперимента: {e}")
            raise e