import pathlib

curr_dir = pathlib.Path(__file__).parent
logs_dir = (curr_dir / "logs").resolve().absolute()

task_params = {
    "question_type": ["control", "payload"],
    "include_trigger": [True, False],
}
models = ["openai/gpt-3.5-turbo", "openai/gpt-4", "openai/gpt-4o"]