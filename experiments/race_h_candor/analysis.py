from inspect_ai.log import read_eval_log, EvalLog, EvalLogInfo, list_eval_logs
from typing import NamedTuple

from experiments.race_h_candor.utils import logs_dir, task_params, models

if __name__ == "__main__":

    log_list = list_eval_logs(str(logs_dir))
    logs: list[EvalLog] = [read_eval_log(log) for log in log_list]

    print(logs[0].eval.task_args)

    # TODO: filter logs by task args and model, and calculate accuracy per task
    # TODO: Plot the candor score as [payload accuracy, no trigger] - [payload accuracy, trigger]
    # TODO: Plot the control score as [control accuracy, no trigger] - [control accuracy, trigger]