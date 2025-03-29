from collections.abc import Generator
from typing import Optional, Union
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool


class AihubmixLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    AIHubMix大语言模型类
    
    这个类通过OpenAI兼容API接口与AIHubMix服务通信。AIHubMix提供了多种模型（如Gemini、Claude、DeepSeek等）
    的访问，并且所有这些模型都已经被转换为与OpenAI兼容的API格式。
    """
    
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        调用模型执行推理
        
        :param model: 模型名称
        :param credentials: 认证信息
        :param prompt_messages: 提示消息列表
        :param model_parameters: 模型参数
        :param tools: 工具列表
        :param stop: 停止词列表
        :param stream: 是否流式输出
        :param user: 用户标识
        :return: 推理结果
        """
        self._add_custom_parameters(credentials)
        return super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        验证认证信息
        
        :param model: 模型名称
        :param credentials: 认证信息
        """
        self._add_custom_parameters(credentials)
        super().validate_credentials(model, credentials)

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        """
        添加自定义参数
        
        :param credentials: 认证信息
        """
        credentials["mode"] = "chat"
        credentials["endpoint_url"] = "https://aihubmix.com/v1"

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        """
        获取可自定义模型的模式
        
        :param model: 模型名称
        :param credentials: 认证信息
        :return: 模型实体
        """
        # 确定模型特性
        features = []
        # 如果支持函数调用，添加工具调用特性
        if credentials.get("function_calling_type") == "function_call":
            features.extend([ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL, ModelFeature.STREAM_TOOL_CALL])
        
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=features,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", 8000)),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=[
                ParameterRule(
                    name="temperature",
                    use_template="temperature",
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="max_tokens",
                    use_template="max_tokens",
                    default=4096,
                    min=1,
                    max=int(credentials.get("max_tokens", 16384)),
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                    type=ParameterType.INT,
                ),
                ParameterRule(
                    name="top_p",
                    use_template="top_p",
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="top_k",
                    use_template="top_k",
                    label=I18nObject(en_US="Top K", zh_Hans="Top K"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="frequency_penalty",
                    use_template="frequency_penalty",
                    label=I18nObject(en_US="Frequency Penalty", zh_Hans="重复惩罚"),
                    type=ParameterType.FLOAT,
                ),
            ],
        ) 