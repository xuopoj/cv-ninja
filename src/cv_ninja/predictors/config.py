"""Configuration management for prediction API credentials."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


class PredictionConfig:
    """Manage prediction API configuration from .env file and CLI options.

    Supports hybrid configuration:
    - .env file: Contains credentials (username, password, API keys, IAM settings)
    - YAML file: Contains endpoint profiles (URLs, modes, etc.)
    - CLI options: Override both .env and YAML settings
    """

    def __init__(
        self,
        env_file: Optional[str] = None,
        profile: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        """Initialize configuration.

        Args:
            env_file: Path to .env file (default: searches for .env in current dir and parents)
            profile: Profile name to load from YAML config (e.g., 'prod', 'test')
            config_file: Path to YAML config file (default: cv-ninja.yaml or endpoints.yaml)
        """
        # Load .env file first (credentials)
        if env_file:
            load_dotenv(env_file)
        else:
            # Search for .env in current directory and parents
            load_dotenv(dotenv_path=self._find_dotenv())

        # Load profile from YAML config
        self.profile_config: Dict[str, Any] = {}
        if profile:
            self.profile_config = self._load_profile(profile, config_file)

    def _find_dotenv(self) -> Optional[Path]:
        """Find .env file in current directory or parents.

        Returns:
            Path to .env file or None if not found
        """
        current = Path.cwd()
        while current != current.parent:
            env_path = current / ".env"
            if env_path.exists():
                return env_path
            current = current.parent
        return None

    def _find_config_file(self) -> Optional[Path]:
        """Find YAML config file in current directory or parents.

        Searches for cv-ninja.yaml or endpoints.yaml.

        Returns:
            Path to config file or None if not found
        """
        current = Path.cwd()
        while current != current.parent:
            for filename in ["cv-ninja.yaml", "endpoints.yaml", "cv-ninja.yml", "endpoints.yml"]:
                config_path = current / filename
                if config_path.exists():
                    return config_path
            current = current.parent
        return None

    def _load_profile(self, profile: str, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load profile configuration from YAML file.

        Args:
            profile: Profile name to load
            config_file: Path to YAML config file (optional)

        Returns:
            Profile configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            KeyError: If profile not found in config
        """
        # Find config file
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self._find_config_file()

        if not config_path or not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found. Create cv-ninja.yaml or endpoints.yaml, "
                f"or specify --config-file"
            )

        # Load YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Get profile
        if 'endpoints' not in config_data:
            raise KeyError("Config file must contain 'endpoints' section")

        if profile not in config_data['endpoints']:
            available = ', '.join(config_data['endpoints'].keys())
            raise KeyError(
                f"Profile '{profile}' not found in config. "
                f"Available profiles: {available}"
            )

        return config_data['endpoints'][profile]

    def get(self, key: str, cli_value: Optional[str] = None, default: Optional[str] = None, profile_key: Optional[str] = None) -> Optional[str]:
        """Get configuration value with precedence: CLI > Profile > .env > default.

        Args:
            key: Environment variable key
            cli_value: Value from CLI option (highest priority)
            default: Default value if not found
            profile_key: Key name in profile config (if different from env var key)

        Returns:
            Configuration value or None
        """
        # CLI option has highest priority
        if cli_value is not None:
            return cli_value

        # Then check profile configuration
        if profile_key and profile_key in self.profile_config:
            return str(self.profile_config[profile_key])

        # Then check environment variable (from .env or system)
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value

        # Finally, return default
        return default

    def get_api_url(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get API URL from config.

        Args:
            cli_value: Value from CLI --api-url option

        Returns:
            API URL
        """
        return self.get("PREDICTION_API_URL", cli_value, profile_key="api_url")

    def get_api_key(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get API key from config.

        Args:
            cli_value: Value from CLI --api-key option

        Returns:
            API key
        """
        return self.get("PREDICTION_API_KEY", cli_value)

    def get_iam_url(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get IAM URL from config.

        Args:
            cli_value: Value from CLI --iam-url option

        Returns:
            IAM URL
        """
        return self.get("PREDICTION_IAM_URL", cli_value, profile_key="iam_url")

    def get_username(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get username from config.

        Args:
            cli_value: Value from CLI --username option

        Returns:
            Username
        """
        return self.get("PREDICTION_USERNAME", cli_value)

    def get_password(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get password from config.

        Args:
            cli_value: Value from CLI --password option

        Returns:
            Password
        """
        return self.get("PREDICTION_PASSWORD", cli_value)

    def get_iam_domain(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get IAM domain from config.

        Args:
            cli_value: Value from CLI --iam-domain option

        Returns:
            IAM domain
        """
        return self.get("PREDICTION_IAM_DOMAIN", cli_value, profile_key="iam_domain")

    def get_iam_project(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get IAM project from config.

        Args:
            cli_value: Value from CLI --iam-project option

        Returns:
            IAM project
        """
        return self.get("PREDICTION_IAM_PROJECT", cli_value, profile_key="iam_project")

    def get_mode(self) -> Optional[str]:
        """Get upload mode from profile config.

        Returns:
            Upload mode ('binary' or 'formdata')
        """
        return self.profile_config.get("mode")

    def get_endpoint(self) -> Optional[str]:
        """Get endpoint path from profile config.

        Returns:
            Endpoint path (e.g., '/upload')
        """
        return self.profile_config.get("endpoint")

    def get_auth_type(self) -> Optional[str]:
        """Get authentication type from profile config.

        Returns:
            Auth type ('api_key' or 'iam')
        """
        return self.profile_config.get("auth_type")
