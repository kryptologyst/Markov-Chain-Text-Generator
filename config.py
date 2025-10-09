"""
Configuration management and environment setup for Markov Chain Text Generator
Handles environment variables, configuration files, and application settings.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Environment(Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "markov_db"
    user: str = "markov_user"
    password: str = "markov_password"
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class MarkovConfig:
    """Markov chain configuration."""
    default_order: int = 2
    default_max_length: int = 100
    default_min_length: int = 10
    default_smoothing: str = "laplace"
    max_order: int = 5
    max_length: int = 500
    cache_models: bool = True
    model_cache_size: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    secret_key: str = "your-secret-key-here"
    data_dir: str = "data"
    models_dir: str = "models"
    static_dir: str = "static"
    templates_dir: str = "templates"
    
    # Sub-configurations
    database: DatabaseConfig = None
    server: ServerConfig = None
    markov: MarkovConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.server is None:
            self.server = ServerConfig()
        if self.markov is None:
            self.markov = MarkovConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """Manages application configuration with environment variable support."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        config_data = {}
        
        # Load from file if provided
        if self.config_file and Path(self.config_file).exists():
            config_data = self._load_from_file(self.config_file)
        
        # Override with environment variables
        env_config = self._load_from_env()
        config_data.update(env_config)
        
        # Create configuration object
        return self._create_config_object(config_data)
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'MARKOV_ENVIRONMENT': 'environment',
            'MARKOV_DEBUG': ('debug', bool),
            'MARKOV_SECRET_KEY': 'secret_key',
            'MARKOV_DATA_DIR': 'data_dir',
            'MARKOV_MODELS_DIR': 'models_dir',
            'MARKOV_STATIC_DIR': 'static_dir',
            'MARKOV_TEMPLATES_DIR': 'templates_dir',
            
            # Server config
            'MARKOV_HOST': 'server.host',
            'MARKOV_PORT': ('server.port', int),
            'MARKOV_WORKERS': ('server.workers', int),
            'MARKOV_RELOAD': ('server.reload', bool),
            'MARKOV_LOG_LEVEL': 'server.log_level',
            
            # Database config
            'MARKOV_DB_HOST': 'database.host',
            'MARKOV_DB_PORT': ('database.port', int),
            'MARKOV_DB_NAME': 'database.name',
            'MARKOV_DB_USER': 'database.user',
            'MARKOV_DB_PASSWORD': 'database.password',
            'MARKOV_DB_POOL_SIZE': ('database.pool_size', int),
            'MARKOV_DB_MAX_OVERFLOW': ('database.max_overflow', int),
            
            # Markov config
            'MARKOV_DEFAULT_ORDER': ('markov.default_order', int),
            'MARKOV_DEFAULT_MAX_LENGTH': ('markov.default_max_length', int),
            'MARKOV_DEFAULT_MIN_LENGTH': ('markov.default_min_length', int),
            'MARKOV_DEFAULT_SMOOTHING': 'markov.default_smoothing',
            'MARKOV_MAX_ORDER': ('markov.max_order', int),
            'MARKOV_MAX_LENGTH': ('markov.max_length', int),
            'MARKOV_CACHE_MODELS': ('markov.cache_models', bool),
            'MARKOV_MODEL_CACHE_SIZE': ('markov.model_cache_size', int),
            
            # Logging config
            'MARKOV_LOG_LEVEL': 'logging.level',
            'MARKOV_LOG_FORMAT': 'logging.format',
            'MARKOV_LOG_FILE': 'logging.file_path',
            'MARKOV_LOG_MAX_SIZE': ('logging.max_file_size', int),
            'MARKOV_LOG_BACKUP_COUNT': ('logging.backup_count', int),
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    key, type_func = config_key
                    value = type_func(value)
                else:
                    key = config_key
                
                # Handle nested keys
                if '.' in key:
                    self._set_nested_key(env_config, key, value)
                else:
                    env_config[key] = value
        
        return env_config
    
    def _set_nested_key(self, config_dict: Dict, key: str, value: Any) -> None:
        """Set nested dictionary key."""
        keys = key.split('.')
        current = config_dict
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> AppConfig:
        """Create AppConfig object from dictionary."""
        # Handle environment string conversion
        if 'environment' in config_data and isinstance(config_data['environment'], str):
            config_data['environment'] = Environment(config_data['environment'])
        
        # Create sub-configurations
        database_config = DatabaseConfig(**config_data.get('database', {}))
        server_config = ServerConfig(**config_data.get('server', {}))
        markov_config = MarkovConfig(**config_data.get('markov', {}))
        logging_config = LoggingConfig(**config_data.get('logging', {}))
        
        # Remove sub-configs from main config
        main_config = {k: v for k, v in config_data.items() 
                      if k not in ['database', 'server', 'markov', 'logging']}
        
        return AppConfig(
            database=database_config,
            server=server_config,
            markov=markov_config,
            logging=logging_config,
            **main_config
        )
    
    def save_config(self, file_path: str) -> None:
        """Save current configuration to file."""
        config_dict = self._config_to_dict()
        
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        config_dict = asdict(self.config)
        
        # Convert enum to string
        if isinstance(config_dict.get('environment'), Environment):
            config_dict['environment'] = config_dict['environment'].value
        
        return config_dict
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def reload_config(self) -> None:
        """Reload configuration from file and environment."""
        self.config = self._load_config()
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.static_dir,
            self.config.templates_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate server config
        if self.config.server.port < 1 or self.config.server.port > 65535:
            errors.append("Server port must be between 1 and 65535")
        
        if self.config.server.workers < 1:
            errors.append("Server workers must be at least 1")
        
        # Validate Markov config
        if self.config.markov.default_order < 1:
            errors.append("Default Markov order must be at least 1")
        
        if self.config.markov.max_order < self.config.markov.default_order:
            errors.append("Max Markov order must be >= default order")
        
        if self.config.markov.default_max_length < self.config.markov.default_min_length:
            errors.append("Default max length must be >= default min length")
        
        if self.config.markov.default_smoothing not in ['laplace', 'good_turing', 'none']:
            errors.append("Invalid smoothing method")
        
        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level. Must be one of: {valid_log_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True


def create_default_config_file(file_path: str = "config.yaml") -> None:
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.save_config(file_path)
    print(f"Default configuration saved to {file_path}")


def load_config(config_file: Optional[str] = None) -> AppConfig:
    """Load configuration from file and environment variables."""
    config_manager = ConfigManager(config_file)
    config_manager.setup_directories()
    config_manager.validate_config()
    return config_manager.get_config()


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    
    # Load and display config
    config = load_config()
    print("Current configuration:")
    print(f"Environment: {config.environment.value}")
    print(f"Debug: {config.debug}")
    print(f"Server: {config.server.host}:{config.server.port}")
    print(f"Data directory: {config.data_dir}")
    print(f"Default Markov order: {config.markov.default_order}")
    print(f"Default smoothing: {config.markov.default_smoothing}")
