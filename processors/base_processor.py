# processors/base_processor.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

# Use TYPE_CHECKING to prevent circular imports at runtime
if TYPE_CHECKING:
    from config.config_manager import ConfigManager
    from utils.display_manager import DisplayManager
    from utils.memory_monitor import MemoryMonitor

class BaseProcessor:
    """
    An abstract base class for all AI model processors (VAD, Diarisation, etc.).

    It provides a common structure for initialising, loading models, unloading models,
    and processing data, ensuring consistency across the pipeline.
    """
    def __init__(
        self,
        config: ConfigManager,
        display: DisplayManager,
        memory_monitor: MemoryMonitor,
        device: torch.device,
        processor_name: str,
    ) -> None:
        """
        Initialises the BaseProcessor.

        Args:
            config: The application's ConfigManager instance.
            display: The application's DisplayManager instance.
            memory_monitor: The application's MemoryMonitor instance.
            device: The torch device (CPU or CUDA) to run the model on.
            processor_name: The name of the specific processor for logging purposes.
        """
        self.config = config
        self.display = display
        self.memory_monitor = memory_monitor
        self.device = device
        self.model: Optional[Any] = None # Generic attribute to hold the loaded model
        self.logger = logging.getLogger(f"{__name__}.{processor_name}")
        self.logger.info(f"'{processor_name}' initialised on device '{device.type}'.")

    def load_model(self) -> None:
        """
        Abstract method for loading the specific AI model into memory.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'load_model' method.")

    def unload_model(self) -> None:
        """
        Unloads the model and clears GPU cache to free up VRAM.
        """
        if self.model is not None:
            model_name = self.model.__class__.__name__
            self.logger.info(f"Unloading model: {model_name}")
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared after model unload.")
        else:
            self.logger.info("No model was loaded, nothing to unload.")

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for running the model's specific processing task.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'process' method.")

