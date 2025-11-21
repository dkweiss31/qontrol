import copy

from jax import Array


class OptimizerRecorder:
    """Records optimization across epochs."""

    def __init__(self, parameters: Array | dict):
        self.total_costs = []
        self.cost_values = []
        self.epoch_times = []
        if isinstance(parameters, dict):
            init_saved_parameters = {key: [] for key in parameters}
        else:
            init_saved_parameters = []
        self.init_saved_parameters = init_saved_parameters
        self.parameters_since_last_save = copy.deepcopy(init_saved_parameters)
        self.current_parameters = parameters
        self.previous_parameters = None
        self.last_save_epoch = -1

    def _append_parameters(self, parameters: Array | dict):
        if isinstance(parameters, dict):
            for key, val in parameters.items():
                self.parameters_since_last_save[key].append(val)
        else:
            self.parameters_since_last_save.append(parameters)

    def record_epoch(
        self,
        parameters: Array | dict,
        cost_values: Array,
        elapsed_time: float,
        total_cost: Array,
    ):
        """Record results from an epoch."""
        self.current_parameters = parameters
        self._append_parameters(parameters)
        self.total_costs.append(total_cost)
        self.cost_values.append(cost_values)
        self.epoch_times.append(elapsed_time)

    def reset(self, epoch: int):
        """Reset saved parameters after a save operation."""
        self.parameters_since_last_save = copy.deepcopy(self.init_saved_parameters)
        self.last_save_epoch = epoch

    def data_to_save(self) -> dict:
        # don't want to resave data from the epoch we last saved at, so +1
        data_dict = {
            'cost_values': self.cost_values[self.last_save_epoch + 1 :],
            'total_cost': self.total_costs[self.last_save_epoch + 1 :],
        }
        if isinstance(self.parameters_since_last_save, dict):
            data_dict |= self.parameters_since_last_save
        else:
            data_dict['parameters'] = self.parameters_since_last_save
        return data_dict
