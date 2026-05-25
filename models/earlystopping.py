import numpy as np
import torch.nn as nn
import copy
import torch


class EarlyStopper:
    """
    EarlyStopper is a utility class to implement early stopping in training neural networks.

    Early stopping is a technique to stop training once the model performance stops improving on a
    held out validation dataset.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state = None
        self.is_parameter = False # To track if best_state is a parameter
        self.counter = 0

    def _save_state(self, obj):
            """Save state for nn.Module, nn.Parameter, or collections of them."""
            # Handle dict (for multi-parameter betting model)
            if isinstance(obj, dict):
                return {k: self._save_state(v) for k, v in obj.items()}

            if isinstance(obj, nn.Parameter):
                return obj.detach().clone()
            elif hasattr(obj, 'state_dict'):
                return copy.deepcopy(obj.state_dict())
            else:
                return copy.deepcopy(obj)

    def _load_state(self, obj, state):
        """Restore state for nn.Module, nn.Parameter, or collections of them."""
        # Handle dict (for multi-parameter betting model)
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._load_state(v, state[k])
        
        # Restore Tensors/Parameters using in-place copy
        elif isinstance(obj, (nn.Parameter, torch.Tensor)):
            with torch.no_grad():
                obj.copy_(state)
        
        # Restore Modules
        elif hasattr(obj, 'load_state_dict'):
            obj.load_state_dict(state)
        

    def early_stop(self, val_loss, model):
        """Return True → stop training."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = self._save_state(model)
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            self._load_state(model, self.best_state)
        else:
            print("EarlyStopper: No best state saved.")

    def reset(self):
        """
        Resets the counter to 0.

        This can be useful when using the same EarlyStopper object across different training phases or model.
        """

        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
