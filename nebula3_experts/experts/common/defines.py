import enum

class ExpertStatus(enum.Enum):
    STARTING = 1
    ACTIVE = 2
    RUNNING = 2

class ExpertCommands(enum.Enum):
    CLI_PREDICT = 1
    PIPELINE_NOTIFY = 2

class OutputStyle(enum.Enum):
    JSON = 1
    DB = 2
    DB_JSON = 3

class RunMode(enum.Enum):
    SERVICE = 'service'
    BATCH = 'batch'
    TASK = 'task'

    def resolve_run_mode(mode: str):
        if mode == RunMode.BATCH.value:
            return RunMode.BATCH
        elif mode == RunMode.TASK.value:
            return RunMode.TASK
        else:
            return RunMode.SERVICE

class ExpertTask:
    id: str
    info: dict
