from comet_ml import Experiment

class CometLogger:
    def __init__(self, api_key: str, project_name: str, experiment_name: str = None):
        self.experiment = Experiment(api_key=api_key, project_name=project_name)
        if experiment_name:
            self.experiment.set_name(experiment_name)

    def log_metrics(self, metrics: dict, step: int):
        self.experiment.log_metrics(metrics, step=step)

    def log_parameters(self, params: dict):
        self.experiment.log_parameters(params)

    def log_image(self, image, name: str, step: int):
        self.experiment.log_image(image, name=name, step=step)

    def end(self):
        self.experiment.end()