import openai
from httpx import Timeout
from typing import Optional

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from .constants import AZURE_OPENAI_API_VERSION


class _CommonAzureOpenAI:
    @staticmethod
    def _get_api_key_from_key_vault(key_vault_url: str, secret_name: str) -> str:
        """从Azure Key Vault获取API密钥"""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=key_vault_url, credential=credential)
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            raise Exception(f"Failed to get secret from Key Vault: {str(e)}")

    @staticmethod
    def _to_credential_kwargs(credentials: dict) -> dict:
        api_version = credentials.get("openai_api_version", AZURE_OPENAI_API_VERSION)
        auth_type = credentials.get("auth_type", "api_key")
        
        credentials_kwargs = {
            "azure_endpoint": credentials["openai_api_base"],
            "api_version": api_version,
            "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }

        if auth_type == "key_vault":
            # 从Key Vault获取API密钥
            key_vault_url = credentials["key_vault_url"]
            secret_name = credentials.get("secret_name", "AzureOpenAIKey")
            api_key = _CommonAzureOpenAI._get_api_key_from_key_vault(key_vault_url, secret_name)
            credentials_kwargs["api_key"] = api_key
        elif auth_type == "managed_identity":
            # 使用Azure托管身份
            from azure.identity import ManagedIdentityCredential
            credential = ManagedIdentityCredential(
                client_id=credentials.get("client_id")
            )
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            credentials_kwargs["azure_ad_token"] = token.token
        else:
            # 传统API Key方式
            credentials_kwargs["api_key"] = credentials["openai_api_key"]

        return credentials_kwargs

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [openai.APIConnectionError, openai.APITimeoutError],
            InvokeServerUnavailableError: [openai.InternalServerError],
            InvokeRateLimitError: [openai.RateLimitError],
            InvokeAuthorizationError: [
                openai.AuthenticationError,
                openai.PermissionDeniedError,
            ],
            InvokeBadRequestError: [
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ],
        }
