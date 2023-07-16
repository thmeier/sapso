import numpy as np

from . import base

class FunctionCallCounterWrapper(object):
    """
    Wrapper class that counts the number of function calls of an optimizer 
    extending base.OptimizationMethod.

    TODO(low): clean up code

    c.f. https://stackoverflow.com/questions/1466676/create-a-wrapper-class-to-call-a-pre-and-post-function-around-existing-functions
    """
    def __init__(self, wrapped_class, *args, **kwargs):

        # TODO: make wrapped_class not dependent on mro which only comes by extending abc.ABC, works for the moment
        assert issubclass(wrapped_class.mro()[0], base.OptimizationMethod), (
            f'Cannot wrap classes other than ones extending '
            f'`sapso.base.OptimizationMethod`. Got `{wrapped_class.__class__}`'
        )

        self.total_function_calls = 0

        objective_orig, *args_remaining = args

        def objective_hooked(*args, **kwargs):
            self.total_function_calls += 1
            return objective_orig(*args, **kwargs)

        self.wrapped_class = wrapped_class(
            objective_hooked, *args_remaining, **kwargs
        )

        _reset_history_orig = self.wrapped_class._reset_history
        def _reset_history_hooked(*args, **kwargs):
            _reset_history_orig(*args, **kwargs)
            self.wrapped_class.history['function_calls'] = list()
        self.wrapped_class._reset_history = _reset_history_hooked

        _update_history_orig = self.wrapped_class._update_history
        def _update_history_hooked(*args, **kwargs):
            _update_history_orig(*args, **kwargs)
            self.wrapped_class.history['function_calls'].append(self.total_function_calls)
        self.wrapped_class._update_history = _update_history_hooked

        _finalize_history_orig = self.wrapped_class._finalize_history
        def _finalize_history_hooked(*args, **kwargs):
            fncalls = self.wrapped_class.history['function_calls']
            _finalize_history_orig(*args, **kwargs)
            self.wrapped_class.history['function_calls'] = np.array(fncalls)
        self.wrapped_class._finalize_history = _finalize_history_hooked

    def __getattr__(self, attr):
            return self.wrapped_class.__getattribute__(attr)
