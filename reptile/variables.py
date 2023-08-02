import copy

__all__ = ['VariableState', 'average_vars', 'interpolate_vars']


class VariableState:
    def __init__(self, model, optimizer=None, trainable: bool = True):
        self._model = model
        self._optimizer = optimizer
        self._trainable = trainable

    def export_variables(self):
        if self._trainable:
            return [v.clone().detach() for v in self._model.parameters()]

        if self._optimizer is None:
            return copy.deepcopy(self._model.state_dict())
        else:
            return {
                'model': copy.deepcopy(self._model.state_dict()),
                'optimizer': copy.deepcopy(self._optimizer.state_dict())
            }

    def import_variables(self, states):
        if self._trainable:
            for state, variable in zip(states, self._model.parameters()):
                variable.data.copy_(state)
            return

        if self._optimizer is None:
            self._model.load_state_dict(states)
        else:
            self._model.load_state_dict(states['model'])
            self._optimizer.load_state_dict(states['optimizer'])


def average_vars(mean_vars, new_vars, num):
    if num == 1:
        return new_vars
    else:
        for mean, new in zip(mean_vars, new_vars):
            mean.mul_(num - 1).add_(new).div_(num)
    return mean_vars

# def average_vars(var_seqs):
#     res = []
#     for var_tuple in zip(*var_seqs):
#         mean = torch.stack(var_tuple).mean(dim=0)
#         res.append(mean)
#     return res


def subtract_vars(vars_one, vars_two):
    assert len(vars_one) == len(vars_two)
    return [v1 - v2 for v1, v2 in zip(vars_one, vars_two)]


def add_vars(vars_one, vars_two):
    assert len(vars_one) == len(vars_two)
    return [v1 + v2 for v1, v2 in zip(vars_one, vars_two)]


def scale_vars(_vars, scale):
    return [v * scale for v in _vars]


def interpolate_vars(old_vars, new_vars, epsilon):
    assert len(old_vars) == len(new_vars)
    return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))
