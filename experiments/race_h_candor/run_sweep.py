import pathlib

from dotenv import load_dotenv
from itertools import product 

from inspect_ai import eval
from candor_bench.tasks.race_h import race_h_candor
from experiments.race_h_candor.utils import logs_dir, task_params, models

if __name__ == "__main__":
    load_dotenv()
    task_grid = list(product(*task_params.values()))

    curr_dir = pathlib.Path(__file__).parent
    logs_dir = (curr_dir / "logs").resolve().absolute()

    # TODO: We need to be able to filter individual runs by their metadata...
    logs = eval(
        [
            race_h_candor(include_trigger=include_trigger, question_type=question_type) 
            for question_type, include_trigger in task_grid
        ],
        model = models,
        limit=10,
        log_dir = str(logs_dir),
    )
