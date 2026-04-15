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
            
            safe_source = truncate(source, 1500)
            safe_ref = truncate(reference, 500)
            safe_a = truncate(model_a_text, 500)
            safe_b = truncate(model_b_text, 500)
            
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
                temperature=0.1,
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
