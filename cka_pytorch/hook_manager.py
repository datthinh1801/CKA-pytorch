from __future__ import annotations

from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn


class HookManager:
    """
    A class to manage forward hooks in a PyTorch model for extracting intermediate features
    (activations) from specified layers.

    This manager allows attaching hooks to specific modules within a `torch.nn.Module`
    and collecting their outputs during the forward pass. It also provides utilities
    to clear collected features and remove hooks.
    """

    def __init__(self, model: nn.Module, layers: List[str]) -> None:
        """
        Initializes the HookManager and registers forward hooks on the specified layers.

        Args:
            model: The PyTorch model (`torch.nn.Module`) to which hooks will be attached.
            layers: A list of strings, where each string is the fully qualified name of a
                    module (layer) within the `model` from which to extract features.
                    These names typically come from `model.named_modules()`.

        Raises:
            ValueError: If no valid layers are found in the model based on the provided `layers` list.
        """
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        # Use list(dict.fromkeys(layers)) to preserve order while removing duplicate layer names
        self.module_names = self._insert_hooks(list(dict.fromkeys(layers)))

    def _hook(
        self,
        module_name: str,
        module: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """
        The hook function that is registered to a module's forward pass.

        This function is called every time the module executes its forward pass.
        It captures the output of the module and stores it in the `self.features` dictionary,
        keyed by the `module_name`.

        Args:
            module_name: The name of the module (layer) to which this hook is attached.
            module: The module itself (unused in this implementation).
            inp: The input tensor(s) to the module (unused in this implementation).
            out: The output tensor(s) from the module's forward pass. This is the activation
                 that will be stored.
        """

    def _insert_hooks(self, layers: List[str]) -> List[str]:
        """
        Registers forward hooks on the specified layers of the model.

        This method iterates through all named modules in the `self.model`.
        If a module's name matches one of the names in the `layers` list,
        a forward hook is registered to that module. The hook will call `_hook`
        to save the module's output. It also keeps track of the `RemovableHandle`
        for each registered hook, allowing them to be removed later.

        Args:
            layers: A list of strings, representing the names of the layers
                    to which hooks should be attached.

        Returns:
            A list of strings containing the names of the layers for which hooks were successfully registered.
            This list might be shorter than the input `layers` if some specified layers were not found.

        Raises:
            ValueError: If, after attempting to register hooks, no layers were found in the model.
                        This typically indicates an issue with the provided layer names.
        """
        filtered_layers: List[str] = []
        for module_name, module in self.model.named_modules():
            if module_name in layers:
                handle = module.register_forward_hook(partial(self._hook, module_name))  # type: ignore
                self.handles.append(handle)
                filtered_layers.append(module_name)

        if len(filtered_layers) != len(layers):
            hooked_set = set(filtered_layers)
            not_hooked = [layer for layer in layers if layer not in hooked_set]
            print(
                f"Warning: Could not find layers: {not_hooked}. They will be ignored."
            )

        if not filtered_layers:
            raise ValueError(
                "No layers were found in the model. Please use `model.named_modules()` "
                "to check the available layer names."
            )

        return filtered_layers

    def clear_features(self) -> None:
        """
        Clears all currently collected features from the `self.features` dictionary.

        This method should be called after processing each batch of data to ensure
        that features from previous batches do not interfere with subsequent calculations.
        """
        self.features = {}

    def clear_all(self) -> None:
        """
        Clears all collected features and removes all registered hooks.

        This method combines the functionality of `clear_features` and `clear_hooks`,
        providing a convenient way to reset the HookManager's state entirely.
        It is useful when you are done with feature extraction for a particular task
        or model and want to free up resources.
        """
        self.clear_hooks()
        self.clear_features()

    def clear_hooks(self) -> None:
        """
        Removes all registered forward hooks from the model.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []
