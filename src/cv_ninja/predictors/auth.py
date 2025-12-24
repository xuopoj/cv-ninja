"""Authentication handlers for prediction APIs."""

import json
import time
from typing import Dict, Optional
import requests


class AuthHandler:
    """Base class for authentication handlers."""

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers.

        Returns:
            Dictionary of headers to add to requests
        """
        raise NotImplementedError


class APIKeyAuth(AuthHandler):
    """Simple API key authentication using Bearer token."""

    def __init__(self, api_key: str):
        """Initialize with API key.

        Args:
            api_key: The API key for authentication
        """
        self.api_key = api_key

    def get_headers(self) -> Dict[str, str]:
        """Get headers with Bearer token.

        Returns:
            Headers with Authorization: Bearer <api_key>
        """
        return {"Authorization": f"Bearer {self.api_key}"}


class IAMTokenAuth(AuthHandler):
    """X-Auth-Token authentication with IAM token acquisition and caching."""

    def __init__(
        self,
        iam_url: str,
        username: str,
        password: str,
        domain: str,
        project: str,
        cache_duration: int = 3600,
    ):
        """Initialize IAM token authentication.

        Args:
            iam_url: URL of the IAM service to get tokens
            username: Username for IAM authentication
            password: Password for IAM authentication
            domain: Domain name for authentication
            project: Project name for authentication scope
            cache_duration: How long to cache token in seconds (default: 3600 = 1 hour)
        """
        self.iam_url = iam_url
        self.username = username
        self.password = password
        self.domain = domain
        self.project = project
        self.cache_duration = cache_duration

        self._token: Optional[str] = None
        self._token_expiry: float = 0

    def get_headers(self) -> Dict[str, str]:
        """Get headers with X-Auth-Token.

        Automatically acquires new token if cached token is expired.

        Returns:
            Headers with X-Auth-Token: <token>

        Raises:
            requests.RequestException: If token acquisition fails
        """
        if self._is_token_expired():
            self._acquire_token()

        return {"X-Auth-Token": self._token}

    def _is_token_expired(self) -> bool:
        """Check if cached token is expired.

        Returns:
            True if token needs to be refreshed
        """
        return time.time() >= self._token_expiry

    def _acquire_token(self) -> None:
        """Acquire new token from IAM service.

        Updates _token and _token_expiry on success.

        Raises:
            requests.RequestException: If token acquisition fails
        """
        payload = json.dumps({
            "auth": {
                "identity": {
                    "methods": [
                        "password"
                    ],
                    "password": {
                        "user": {
                            "name": self.username,
                            "password": self.password,
                            "domain": {
                                "name": self.domain
                            }
                        }
                    }
                },
                "scope": {
                    "project": {
                        "name": self.project
                    }
                }
            }
        })

        response = requests.post(
            self.iam_url,
            data=payload,
            verify=False,
            timeout=10,
        )
        response.raise_for_status()

        # Extract token from response headers
        token = response.headers.get("X-Subject-Token")
        if token:
            self._token = token
            self._token_expiry = time.time() + self.cache_duration
        else:
            raise ValueError("X-Subject-Token not found in IAM response headers")

    def clear_cache(self) -> None:
        """Clear cached token, forcing refresh on next request."""
        self._token = None
        self._token_expiry = 0
