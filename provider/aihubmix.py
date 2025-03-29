import logging
from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class AihubmixProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        验证提供商凭据
        如果验证失败，抛出异常

        :param credentials: 提供商凭据，凭据表单在 `provider_credential_schema` 中定义。
        """
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(model="gemini-2.0-pro-exp-02-05", credentials=credentials)
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(f"{self.get_provider_schema().provider} credentials validate failed")
            raise ex 