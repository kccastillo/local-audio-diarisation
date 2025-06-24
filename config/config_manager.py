# config/config_manager.py

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

# Get a logger for this module
module_logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class ConfigManager:
    """
    A Singleton class to manage loading and accessing configuration from a YAML file.

    This ensures that the configuration is loaded only once and provides
    a consistent access point throughout the application. It also handles
    path resolution and directory creation.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Implements the Singleton pattern to ensure only one instance of ConfigManager exists.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Flag to ensure __init__ runs only once.
            cls._instance._initialised = False
        return cls._instance

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """
        Initialises the ConfigManager instance on its first creation.

        Args:
            config_path: Optional path to the config file. If None, it defaults
                         to 'config/config.yaml' relative to the project root.
        """
        if self._initialised:
            return

        self.logger = logging.getLogger(__name__)

        # Determine the base directory of the project (assuming this file is in config/)
        project_root = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else project_root / "config" / "config.yaml"
        self.config = {}

        try:
            self._load_config()
            self._resolve_paths()
            self._create_directories()
        except Exception as e:
            self.logger.error(f"Failed to initialise ConfigManager: {e}", exc_info=True)
            raise ConfigurationError(f"Configuration initialisation failed: {e}") from e

        self._initialised = True
        self.logger.info("ConfigManager initialised successfully.")

    def _load_config(self) -> None:
        """Loads the configuration from the specified YAML file."""
        self.logger.debug(f"Attempting to load config from: {self.config_path}")
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                # Use the faster C-based loader if available
                if hasattr(yaml, 'CSafeLoader'):
                    self.config = yaml.load(f, Loader=yaml.CSafeLoader) or {}
                else:
                    self.config = yaml.safe_load(f) or {}
                
                if not self.config:
                    self.logger.warning(f"Configuration file is empty: {self.config_path}")

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration file: {e}") from e

    def _resolve_paths(self) -> None:
        """
        Resolves all paths in the 'paths' section of the config, making them
        absolute based on the configured 'base_dir'.
        """
        base_dir_str = self.get('paths', 'base_dir', default='.')
        base_dir = Path(base_dir_str).resolve()
        
        # This relies on the YAML structure being correct (duck typing)
        if 'paths' in self.config and isinstance(self.config['paths'], dict):
            self.config['paths']['base_dir'] = str(base_dir)
            for key, value in self.config['paths'].items():
                if key != 'base_dir':
                    absolute_path = base_dir / str(value)
                    self.config['paths'][key] = str(absolute_path)
        
        if 'auth' in self.config and isinstance(self.config['auth'], dict):
            auth_token_path_str = self.config['auth'].get('token_path')
            if auth_token_path_str:
                self.config['auth']['token_path'] = str(Path(auth_token_path_str).expanduser().resolve())

    def _create_directories(self) -> None:
        """Creates the directories specified in the 'paths' config if they don't exist."""
        paths_to_create = self.get('paths', default={})
        if isinstance(paths_to_create, dict):
            for key, path_str in paths_to_create.items():
                if key != 'base_dir':
                    Path(path_str).mkdir(parents=True, exist_ok=True)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieves a nested configuration value using a sequence of keys.

        Example: config.get('processing', 'whisper', 'model_size', default='base')

        Args:
            *keys: A sequence of keys to navigate the nested dictionary.
            default: The value to return if the key path is not found. Defaults to None.

        Returns:
            The configuration value, or the default value if not found.
        """
        value = self.config
        for key in keys:
            try:
                # Navigate down the dictionary
                value = value[key]
            except (KeyError, TypeError):
                # If a key is not found or if a value is not a dictionary,
                # return the default value.
                return default
        return value

    def get_path(self, key: str) -> Path:
        """
        Retrieves a path from the 'paths' section of the config as a Path object.

        Args:
            key: The key for the path within the 'paths' dictionary.

        Returns:
            A Path object for the requested path.

        Raises:
            ConfigurationError: If the 'paths' section or the specific key is not found.
        """
        path_str = self.get('paths', key)
        if not path_str:
            raise ConfigurationError(f"Path key '{key}' not found in 'paths' section of configuration.")
        return Path(path_str)

    def reload(self) -> None:
        """Reloads the configuration from the file."""
        self.logger.info("Reloading configuration...")
        self._load_config()
        self._resolve_paths()
        self._create_directories()
        self.logger.info("Configuration reloaded successfully.")
