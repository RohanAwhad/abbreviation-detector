import wandb


class WandBLogger:
    def __init__(self, project_name):
        self.project_name = project_name

    def init(self, config, name=None):
        wandb.init(
            project=self.project_name,
            name=name,
            config=config,
        )

    def log(self, data, step):
        wandb.log(data=data, step=step)

    def watch(self, model):
        wandb.watch(model)
