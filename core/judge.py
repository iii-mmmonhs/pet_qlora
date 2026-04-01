import torch
import logging
import random
import json
import os
from huggingface_hub import InferenceClient
from json_repair import repair_json
from config import JUDGE_MODEL, MAX_TOKENS_JUDGE

LLM_TOKEN = os.getenv("LLM_TOKEN")
logger = logging.getLogger(__name__)


class Judge:
    """
    Оценка качества суммаризации с помощью LLM
    
    Использует Hugging Face API для отправки запросов к модели-судье,
    возвращает JSON.
    """
    @staticmethod
    def evaluate(source, reference, model_a_text, model_b_text):

        if not LLM_TOKEN:
            logger.error("LLM_TOKEN не указан")
            return {
                "score_a_faithfulness": 0,
                "score_a_conciseness": 0,
                "score_b_faithfulness": 0,
                "score_b_conciseness": 0,
                "winner": "Error",
                "reason": "LLM_TOKEN missing"
            }
        
        try:
            client = InferenceClient(api_key=LLM_TOKEN)

            def truncate(text, limit=300):
                return (text[:limit] + "...") if len(text) > limit else text
            
            safe_source = truncate(source, 700)
            safe_ref = truncate(reference, 300)
            safe_a = truncate(model_a_text, 300)
            safe_b = truncate(model_b_text, 300)
            
            system_message = (
                "You are an expert linguist specializing in text summarization evaluation. "
                "Your task is to compare two model outputs against a reference summary and the original source text."
            )
            
            user_message = f"""
            Here is the data for evaluation:
            - **Source Text**: {safe_source}
            - **Reference Summary**: {safe_ref}
            - **Model A (QLoRA) Output**: {safe_a}
            - **Model B (Base) Output**: {safe_b}
            
            Please evaluate both models on two criteria (scale 1-5):
            1. **Faithfulness**: How accurately does the summary reflect the source facts?
            2. **Conciseness**: Is the summary concise without losing key information?
            
            Determine the winner (Model A or Model B) based on these scores and provide a brief reason.
            
            Return the result strictly as a JSON object with the following keys:
            - score_a_faithfulness (int)
            - score_a_conciseness (int)
            - score_b_faithfulness (int)
            - score_b_conciseness (int)
            - winner (string: "Model A" or "Model B")
            - reason (string)
            """
            
            logger.info("Отправка запроса к судье")
            response = client.chat_completion(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=MAX_TOKENS_JUDGE,
                temperature=0.0,
                response_format={"type": "json_object"} 
            )

            raw_content = response.choices[0].message.content
                        
            repaired_content = repair_json(raw_content, return_objects=False)
            
            if not repaired_content:
                raise ValueError("json_repair вернул пустую строку")

            result = json.loads(repaired_content)
                        
            logger.info(f"Judge result: {result}")
            return result

        except Exception as e:
            logger.error(f"Judge error: {e}")
            return {
                "score_a_faithfulness": 0,
                "score_a_conciseness": 0,
                "score_b_faithfulness": 0,
                "score_b_conciseness": 0,
                "winner": "Error",
                "reason": f"Error: {str(e)[:50]}"
            }


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
            
        device_map = "auto" if torch.cuda.is_available() else None
            
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