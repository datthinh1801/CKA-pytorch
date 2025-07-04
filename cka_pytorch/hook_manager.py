from __future__ import annotations

from functools import partial
from typing import Dict, List, Type

import torch
import torch.nn as nn
from lightning.fabric.wrappers import _FabricModule

from .utils import gram


class HookManager:
    """
    A class to manage forward hooks in a PyTorch model for extracting intermediate features
    (activations) from specified layers.

    This manager allows attaching hooks to specific modules within a `torch.nn.Module`
    and collecting their outputs during the forward pass. It also provides utilities
    to clear collected features and remove hooks.
    """

    def __init__(
        self,
        model: nn.Module | _FabricModule,
        layers: List[str] | List[Type] | None = None,
        recursive: bool = True,
    ) -> None:
        """
        Initializes the HookManager and registers forward hooks on the specified layers.

        Args:
            model: The PyTorch model (`torch.nn.Module`) to which hooks will be attached.
            layers: A list of strings, where each string is the fully qualified name of a
                    module (layer) within the `model` from which to extract features.
                    These names typically come from `model.named_modules()`.
                    If `None`, all layers will be hooked. Defaults to `None`.
            recursive: A boolean indicating whether to register hooks recursively on the model.
                      If `True`, hooks will be registered on all submodules of the specified layers.

        Raises:
            ValueError: If no valid layers are found in the model based on the provided `layers` list.
        """
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        if layers is None:
            # If no layers are specified, default to hooking all layers
            layers = [nn.Module]

        self.module_names = self._insert_hooks(
            module=model,
            layers=layers,
            recursive=recursive,
        )

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
        del (
            module,
            inp,
        )  # Unused parameters, but kept for compatibility with the hook signature
        batch_size = out.size(0)
        feature = out.reshape(batch_size, -1)
        feature = gram(feature)
        self.features[module_name] = feature

    def _should_hook_module(
        self,
        child: nn.Module,
        module_name: str,
        curr_name: str,
        layers: List[str] | List[Type],
        recursive: bool = True,
    ) -> bool:
        """
        Determines whether a module should be hooked based on the layers criteria.

        Args:
            child: The module to check
            module_name: The short name of the module
            curr_name: The full qualified name of the module
            layers: The list of layer names or types to match against
            recursive: Whether to check recursively in submodules

        Returns:
            True if the module should be hooked, False otherwise
        """
        if not layers:
            return False

        # If the child is a container, we should not hook it directly,
        # but rather check its children recursively.
        # Since the output of the last children of the container is the output of the container,
        # we do not need to hook the container itself.
        if recursive and isinstance(child, (nn.Sequential, nn.ModuleList)):
            return False

        # Check if the module name matches any of the specified layers
        if isinstance(layers[0], str):
            return module_name in layers or curr_name in layers

        # Check if the module is of a specified layer type
        if isinstance(layers[0], Type):
            type_layers = [layer for layer in layers if isinstance(layer, type)]
            return bool(type_layers and isinstance(child, tuple(type_layers)))

        return False

    def _insert_hooks(
        self,
        module: nn.Module | _FabricModule,
        layers: List[str] | List[Type],
        recursive: bool = True,
        prev_name: str = "",
    ) -> List[str]:
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
        if isinstance(module, _FabricModule):
            # If the module is a FabricModule, use its underlying module for hook registration
            module = module.module

        filtered_layers: List[str] = []
        for module_name, child in module.named_children():
            curr_name = f"{prev_name}.{module_name}" if prev_name else module_name
            curr_name = curr_name.replace("_model.", "")
            num_grandchildren = len(list(child.named_children()))

            if recursive and num_grandchildren > 0:
                filtered_layers.extend(
                    self._insert_hooks(
                        module=child,
                        layers=layers,
                        recursive=recursive,
                        prev_name=curr_name,
                    )
                )

            # Check if this module should be hooked
            if self._should_hook_module(
                child, module_name, curr_name, layers, recursive
            ):
                handle = child.register_forward_hook(partial(self._hook, curr_name))  # type: ignore
                self.handles.append(handle)
                filtered_layers.append(curr_name)

        return filtered_layers

    def clear_features(self) -> None:
        """
        Clears all currently collected features from the `self.features` dictionary.

        This method should be called after processing each batch of data to ensure
        that features from previous batches do not interfere with subsequent calculations.
        """
        self.features = {}

    def clear_hooks(self) -> None:
        """
        Removes all registered forward hooks from the model.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

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
