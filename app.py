import gradio as gr
import logging
from core.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s:%(funcName)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

try:
    runner = ExperimentRunner()
    runner.load_resources()
    logger.info("Модели загружены")
except Exception as e:
    logger.error(f"Не удалось загрузить модели {e}")
    runner = None

def run_and_format():
    """
    Запускает эксперимент сравнения моделей и форматирует результаты для вывода в Gradio.
    """
    if runner is None:
        err_msg = "Модели не загрузились при старте"
        return (
            "Ошибка", "",
            "Ошибка", {},
            "Ошибка", {},
            {"error": err_msg},
            f"Ошибка: {err_msg}"
        )

    try:
        res = runner.run()
        return res
    except Exception as e:
        logger.error(f"Ошибка в эксперименте: {e}", exc_info=True)
        err_msg = str(e)
        return (
            "Ошибка", "",
            "Ошибка", {},
            "Ошибка", {},
            {"error": err_msg},
            f"Ошибка: {err_msg}"
        )

# интерфейс Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# QLoRA vs Base Model")
    gr.Markdown("Сравнение качества суммаризации дообученной и базовой Flan T5")
    
    with gr.Row():
        btn_go = gr.Button("Запустить эксперимент", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            src_display = gr.Textbox(label="Исходная статья", lines=6, interactive=False)
            ref_display = gr.Textbox(label="Эталонная суммаризация", lines=3, interactive=False)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("### QLoRA")
            q_out = gr.Textbox(label="Summary", lines=4, interactive=False)
        with gr.Column(variant="panel"):
            gr.Markdown("### Исходная модель")
            b_out = gr.Textbox(label="Summary", lines=4, interactive=False)
            
    with gr.Column(variant="compact", elem_id="judge_area"):
        gr.Markdown("## Вердикт судьи")
        judge_out = gr.JSON(label="Результат оценки")

    btn_go.click(
        fn=run_and_format,
        outputs=[src_display, ref_display, q_out, b_out, judge_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
