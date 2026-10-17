"""把已登录的本地 Codex 适配为 LangChain 结构化聊天模型。"""
import json
import signal
import tempfile
import tomllib
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def strict_schema(value):
    """为 Codex 的结构化输出补齐 object 的 required / additionalProperties。"""
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: strict_schema(item) for key, item in value.items() if key != 'default'}
    if result.get('type') == 'object':
        result['additionalProperties'] = False
        result['required'] = list(result.get('properties', {}))
    return result


def run_codex(prompt, schema, *, model=None, timeout=180,
              instructions='你是图谱抽取器。只依据提供的文本抽取，不推断。'):
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    overrides = ['web_search="disabled"', 'features.shell_tool=false',
                 'features.unified_exec=false', 'features.code_mode=false',
                 'features.multi_agent=false', 'features.apps=false', 'features.memories=false']
    config_path = Path.home() / '.codex/config.toml'
    if config_path.exists():
        config = tomllib.loads(config_path.read_text())
        overrides += [f'mcp_servers.{name}.enabled=false'
                      for name in config.get('mcp_servers', {})]

    def timed_out(_signum, _frame):
        raise TimeoutError(f'Codex 调用超过 {timeout} 秒')

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with tempfile.TemporaryDirectory(prefix='graph-rag-codex-') as cwd:
            sdk_config = CodexConfig(cwd=cwd, config_overrides=tuple(overrides))
            with Codex(sdk_config) as codex:
                thread = codex.thread_start(
                    cwd=cwd, model=model, sandbox=Sandbox.read_only,
                    approval_mode=ApprovalMode.deny_all, ephemeral=True,
                    base_instructions=instructions + '禁止使用工具或读取外部内容。',
                )
                result = thread.run(prompt, output_schema=schema)
                if result.status.value != 'completed' or not result.final_response:
                    raise RuntimeError('Codex 调用未完成或未返回结果')
                kinds = [getattr(getattr(item, 'root', item), 'type', 'unknown')
                         for item in result.items]
                if any(kind not in ('userMessage', 'agentMessage', 'reasoning') for kind in kinds):
                    raise RuntimeError('Codex 调用中发生了额外工具活动，拒绝结果')
                return json.loads(result.final_response)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class CodexChatModel(BaseChatModel):
    """仅提供同步结构化输出；沿用本机 Codex 默认模型和登录。"""
    model: str | None = None
    timeout: int = 180

    @property
    def _llm_type(self):
        return 'codex-local'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError('请通过 with_structured_output 使用此模型')

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        output_schema = strict_schema(schema.model_json_schema())

        def invoke(prompt):
            data = run_codex(prompt.to_string(), output_schema,
                             model=self.model, timeout=self.timeout)
            parsed = schema.model_validate(data)
            if include_raw:
                return {'raw': AIMessage(content=json.dumps(data, ensure_ascii=False)),
                        'parsed': parsed, 'parsing_error': None}
            return parsed

        return RunnableLambda(invoke)
