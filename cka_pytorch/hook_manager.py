from typing import Callable, Optional, Tuple, Type, Union

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from cka_pytorch.utils import gram

_HOOK_LAYER_TYPES = (
    Bottleneck,
    BasicBlock,
    nn.Conv2d,
    nn.AdaptiveAvgPool2d,
    nn.MaxPool2d,
    nn.modules.batchnorm._BatchNorm,
)


class HookManager:
    """
    A class to manage hooks in a PyTorch model for feature extraction.

    This class provides a convenient way to register, manage, and remove hooks
    from a model. It is designed to capture intermediate layer outputs, which
    can then be used for analysis, such as calculating CKA.

    The manager can recursively traverse a model and attach forward hooks to
    layers of specified types. It also provides built-in hook functions for
    common feature transformations like flattening and average pooling.
    """

    def __init__(
        self,
        model: nn.Module,
        hook_fn: Optional[Union[str, Callable]] = None,
        hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES,
        calculate_gram: bool = True,
    ) -> None:
        """
        Initializes the HookManager and registers the hooks.

        Args:
            model: The PyTorch model to attach hooks to.
            hook_fn: The hook function or a string identifier for a built-in function
                     ('flatten' or 'avgpool'). If None, 'flatten' is used.
            hook_layer_types: A tuple of nn.Module layer types to attach hooks to.
            calculate_gram: If True, computes the Gram matrix of the features
                            before storing them.
        """
        self.model = model
        self.hook_fn = hook_fn
        self.hook_layer_types = hook_layer_types
        self.calculate_gram = calculate_gram

        for layer in self.hook_layer_types:
            if not issubclass(layer, nn.Module):
                raise TypeError(f"Class {layer} is not an nn.Module.")

        if self.hook_fn is None:
            self.hook_fn = self.flatten_hook_fn
        elif isinstance(self.hook_fn, str):
            hook_fn_dict = {
                "flatten": self.flatten_hook_fn,
                "avgpool": self.avgpool_hook_fn,
            }
            if self.hook_fn in hook_fn_dict:
                self.hook_fn = hook_fn_dict[self.hook_fn]
            else:
                raise ValueError(
                    f"No hook function named {self.hook_fn}. Options: {list(hook_fn_dict.keys())}"
                )

        self.features: list[torch.Tensor] = []
        self.module_names: list[str] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        self.register_hooks(self.hook_fn)

    def clear_features(self) -> None:
        """
        Clears the collected features and module names.
        """
        self.features = []
        self.module_names = []

    def clear_all(self) -> None:
        """
        Clears all collected data and removes all registered hooks.
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
        for m in self.model.modules():
            if hasattr(m, "module_name"):
                delattr(m, "module_name")

    def register_hooks(self, hook_fn: Callable) -> None:
        """
        Registers hooks to the model recursively.

        Args:
            hook_fn: The hook function to be registered. It must accept
                     (module, input, output) as arguments.
        """
        self._register_hook_recursive(self.model, hook_fn, prev_name="")

    def _register_hook_recursive(
        self, module: nn.Module, hook_fn: Callable, prev_name: str = ""
    ) -> None:
        """
        Recursively traverses the model and registers hooks to children modules.

        Args:
            module: The current module to traverse.
            hook_fn: The hook function to register.
            prev_name: The name of the parent module, used to build the full module name.
        """
        for name, child in module.named_children():
            curr_name = f"{prev_name}.{name}" if prev_name else name
            curr_name = curr_name.replace("_model.", "")
            num_grandchildren = len(list(child.children()))

            if num_grandchildren > 0:
                self._register_hook_recursive(child, hook_fn, prev_name=curr_name)

            if isinstance(child, self.hook_layer_types):
                handle = child.register_forward_hook(hook_fn)
                self.handles.append(handle)
                setattr(child, "module_name", curr_name)

    def flatten_hook_fn(
        self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor
    ) -> None:
        """
        A hook function that flattens the output of a module.

        The flattened feature is stored in `self.features`.

        Args:
            module: The module to which the hook is attached.
            inp: The input to the module (unused).
            out: The output from the module.
        """
        batch_size = out.size(0)
        feature = out.reshape(batch_size, -1)

        if self.calculate_gram:
            feature = gram(feature)

        module_name = getattr(module, "module_name")
        self.features.append(feature)
        self.module_names.append(module_name)

    def avgpool_hook_fn(
        self, module: nn.Module, inp: torch.Tensor, out: torch.Tensor
    ) -> None:
        """
        A hook function that applies average pooling to the output of a module.

        The pooled feature is stored in `self.features`.

        Args:
            module: The module to which the hook is attached.
            inp: The input to the module (unused).
            out: The output from the module.
        """
        if out.dim() == 4:
            feature = out.mean(dim=(-1, -2))
        elif out.dim() == 3:
            feature = out.mean(dim=-1)
        else:
            feature = out

        if self.calculate_gram:
            feature = gram(feature)

        module_name = getattr(module, "module_name")
        self.features.append(feature)
        self.module_names.append(module_name)
