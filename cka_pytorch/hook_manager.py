from __future__ import annotations

from functools import partial
from typing import Dict, List

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
        layers: list[str] | list[type] | None = None,
        recursive: bool = True,
    ) -> None:
        """
        Initializes the HookManager and registers forward hooks on the specified layers.

        Args:
            model (torch.nn.Module | _FabricModule): The module to which hooks will be attached.
            layers (list[str] | list[type] | None): A list of strings (fully qualified module names)
                or a list of types (layer classes) within the model from which to extract features.
                If None, all layers will be hooked. Defaults to None.
            recursive (bool): Whether to register hooks recursively on the model.
                If True, hooks will be registered on all submodules of the specified layers.

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
            module_name (str): The name of the module (layer) to which this hook is attached.
            module (torch.nn.Module): The module itself (unused in this implementation).
            inp (torch.Tensor): The input tensor(s) to the module (unused in this implementation).
            out (torch.Tensor): The output tensor(s) from the module's forward pass. This is the activation that will be stored.
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
        layers: list[str] | list[type],
        recursive: bool = True,
    ) -> bool:
        """
        Determines whether a module should be hooked based on the layers criteria.

        Args:
            child (torch.nn.Module): The module to check.
            module_name (str): The short name of the module.
            curr_name (str): The fully qualified name of the module.
            layers (list[str] | list[type]): The list of layer names (str) or types (type) to match against.
            recursive (bool): For handling Sequential/ModuleList containers. If True, Sequential/ModuleList containers are not hooked directly.

        Returns:
            bool: True if the module should be hooked, False otherwise.
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
        if isinstance(layers[0], type):
            return isinstance(child, tuple(layers))  # type: ignore

        return False

    def _insert_hooks(
        self,
        module: nn.Module | _FabricModule,
        layers: list[str] | list[type],
        recursive: bool = True,
        prev_name: str = "",
    ) -> list[str]:
        """
        Registers forward hooks on the specified layers of the model.

        This method iterates through all named modules in the model.
        If a module's name or type matches one of the entries in the `layers` list,
        a forward hook is registered to that module.
        The behavior for Sequential and ModuleList containers depends on the `recursive` flag:
            - If `recursive=True`, Sequential/ModuleList containers are not hooked,
                but their children are processed recursively.
            - If `recursive=False`, Sequential/ModuleList containers can be hooked if they match,
                but their children are not processed.

        Args:
            module (torch.nn.Module | _FabricModule): The model or submodule to process.
            layers (list[str] | list[type]): List of layer names (str) or types (type) to hook.
            recursive (bool): Whether to recursively process submodules.
            prev_name (str): The qualified name prefix for the current module (used for nested modules).

        Returns:
            list[str]: Names of the layers for which hooks were successfully registered.
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
