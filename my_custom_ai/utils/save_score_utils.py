import wandb

class WeightsAndBiases:
    def __init__(self, logger_init):
        required_keys_wandb = ['entity', 'project', 'name', 'configs']
        # logger_init validation
        self._validation(logger_init, required_keys_wandb)
        # run initialization
        self.run = wandb.init(
            entity= logger_init["entity"],    # your username on WeightsAndBiases ("imangw-florence-university")
            project= logger_init["project"],  # name of your project
            name= logger_init["name"],        # name of your experiment
            config= logger_init["configs"]    # dict with all settings you want to save
                                              # example: config={
                                              #            "learning_rate": lr,
                                              #            "architecture": model_type,
                                              #            "dataset": dataset,
                                              #            "epochs": epochs })
        )

    @staticmethod
    def _validation(logger_init, required_keys_wandb):
        if type(logger_init) != dict:
            raise ValueError(
                "logger_init must be a dictionary as: {'entity': str, 'project': str, 'name':str, 'configs': dict}.")
        else:
            for key in required_keys_wandb:
                if key not in logger_init:
                    raise ValueError(f"logger_init must contain key {key}.")

                if key == 'configs':
                    if type(logger_init[key]) != dict:
                        raise ValueError(f"logger_init['configs'] must be a dictionary.")

    def log_metrics(self, metrics):
        self.run.log(metrics)

    def finish(self, *args, **kwargs):
        self.run.finish(*args, **kwargs)

