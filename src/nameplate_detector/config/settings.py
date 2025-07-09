"""
Configuration Settings for Nameplate Detector

Manages configuration settings using environment variables with sensible defaults.
"""

import os
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache

@dataclass
class Settings:
    """
    Configuration settings for the nameplate detector application.
    """
    
    # Model Configuration
    model_path: str = field(default_factory=lambda: os.getenv('MODEL_PATH', 'models/best_nameplate_classifier.pth'))
    device: str = field(default_factory=lambda: os.getenv('DEVICE', 'auto'))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv('CONFIDENCE_THRESHOLD', '0.7')))
    
    # API Configuration
    api_host: str = field(default_factory=lambda: os.getenv('API_HOST', '0.0.0.0'))
    api_port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
    api_reload: bool = field(default_factory=lambda: os.getenv('API_RELOAD', 'false').lower() == 'true')
    
    # Frontend Configuration
    frontend_host: str = field(default_factory=lambda: os.getenv('FRONTEND_HOST', 'localhost'))
    frontend_port: int = field(default_factory=lambda: int(os.getenv('FRONTEND_PORT', '3000')))
    backend_port: int = field(default_factory=lambda: int(os.getenv('BACKEND_PORT', '3001')))
    
    # Camera Configuration
    camera_device: int = field(default_factory=lambda: int(os.getenv('CAMERA_DEVICE', '0')))
    camera_width: int = field(default_factory=lambda: int(os.getenv('CAMERA_WIDTH', '640')))
    camera_height: int = field(default_factory=lambda: int(os.getenv('CAMERA_HEIGHT', '480')))
    camera_fps: int = field(default_factory=lambda: int(os.getenv('CAMERA_FPS', '30')))
    
    # Detection Configuration
    detection_interval: float = field(default_factory=lambda: float(os.getenv('DETECTION_INTERVAL', '0.1')))
    max_detections: int = field(default_factory=lambda: int(os.getenv('MAX_DETECTIONS', '100')))
    
    # Field Extraction Configuration
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv('MAX_NEW_TOKENS', '200')))
    temperature: float = field(default_factory=lambda: float(os.getenv('TEMPERATURE', '0.7')))
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE'))
    
    # Data Directories
    data_dir: str = field(default_factory=lambda: os.getenv('DATA_DIR', 'data'))
    models_dir: str = field(default_factory=lambda: os.getenv('MODELS_DIR', 'models'))
    logs_dir: str = field(default_factory=lambda: os.getenv('LOGS_DIR', 'logs'))
    
    # Security Settings
    cors_origins: str = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '*'))
    
    def __post_init__(self):
        """Validate and process settings after initialization."""
        # Ensure directories exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
    
    @property
    def model_full_path(self) -> str:
        """Get the full path to the model file."""
        if os.path.isabs(self.model_path):
            return self.model_path
        return os.path.join(self.models_dir, os.path.basename(self.model_path))
    
    @property
    def cors_origins_list(self) -> list:
        """Get CORS origins as a list."""
        if self.cors_origins == '*':
            return ['*']
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    def update_from_dict(self, config_dict: dict):
        """Update settings from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            key: getattr(self, key) 
            for key in self.__dataclass_fields__.keys()
        }

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings instance.
    
    Returns:
        Settings instance (cached)
    """
    return Settings()

def load_settings_from_file(config_file: Union[str, Path]) -> Settings:
    """
    Load settings from a configuration file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Settings instance
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    settings = Settings()
    
    # Load from different file formats
    if config_file.suffix == '.json':
        import json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        settings.update_from_dict(config_dict)
    
    elif config_file.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            settings.update_from_dict(config_dict)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    
    elif config_file.suffix == '.env':
        # Load environment variables from .env file
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        settings = Settings()  # Reload with new environment variables
    
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    return settings

def create_config_file(config_file: Union[str, Path], settings: Settings = None):
    """
    Create a configuration file with current settings.
    
    Args:
        config_file: Path to create the configuration file
        settings: Settings instance to save (uses current settings if None)
    """
    config_file = Path(config_file)
    settings = settings or get_settings()
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    if config_file.suffix == '.json':
        import json
        with open(config_file, 'w') as f:
            json.dump(settings.to_dict(), f, indent=2)
    
    elif config_file.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_file, 'w') as f:
                yaml.safe_dump(settings.to_dict(), f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configuration files")
    
    elif config_file.suffix == '.env':
        with open(config_file, 'w') as f:
            f.write("# Nameplate Detector Configuration\n\n")
            for key, value in settings.to_dict().items():
                f.write(f"{key.upper()}={value}\n")
    
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}") 