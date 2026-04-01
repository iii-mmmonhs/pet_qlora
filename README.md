# QLoRA Fine-Tuning of FLAN-T5 for Summarization + LLM-as-a-Judge
Проект представляет собой дообучение модели FLAN-T5-Base на датасете CNN с использованием квантования и низкоранговой адаптации - QLoRA.
### Описание дообучения
- Для дообучения выбрана модель Т5, поскольку для задачи суммаризации требуется архитектура encoder-decoder.
- Использовано QLoRA (4-bit quantization + LoRA).
- Метрики: ROUGE-1, ROUGE-2, ROUGE-L.
### Стек
Python, PyTorch, HF Transformers, PEFT, Datasets, Gradio.

```/
├── data/                            # файлы адаптеров
│   ├── adapter_config.json			 
│   └── adapter_model.safetensors	  
├── train                  		       # код для дообучения
├── config.py                        # гиперпараметры и конфигурации
├── app.py                           # основная логика
└── core                  		       # вспомогательные функции
    ├── judge.py                     # класс судьи
    └── runner.py                    # управление экспериментами

```
