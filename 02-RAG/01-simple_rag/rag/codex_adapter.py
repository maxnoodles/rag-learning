"""通过官方 Python SDK 调用已登录的本地 Codex，只生成答案。"""
import json
import re
import signal
import tempfile
import tomllib
from pathlib import Path


def generate(prompt, valid_citations, model=None, timeout=300, codex_bin=None):
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
    # 默认使用 SDK 配套的固定版本运行时；仅在显式指定时使用其他 CLI。
    binary = codex_bin
    overrides = ['web_search="disabled"', 'features.shell_tool=false',
                 'features.unified_exec=false', 'features.code_mode=false',
                 'features.multi_agent=false', 'features.apps=false', 'features.memories=false']
    # 禁用用户已配置的 MCP，防止生成阶段另行搜索知识库或外部数据。
    config_file = Path.home()/'.codex/config.toml'
    if config_file.exists():
        config = tomllib.loads(config_file.read_text())
        overrides += [f'mcp_servers.{name}.enabled=false'
                      for name in config.get('mcp_servers', {})]
    schema = {'type':'object','additionalProperties':False,
              'properties':{'status':{'type':'string','enum':['answered','insufficient','conflict']},
                            'answer':{'type':'string'},
                            'citations':{'type':'array','items':{'type':'string'}}},
              'required':['status','answer','citations']}

    def timed_out(_signum, _frame):
        raise TimeoutError(f'Codex 生成超过 {timeout} 秒；检索结果已保留')

    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        # 避免本项目 AGENTS.md 或文件内容被额外加载为生成阶段的知识来源。
        with tempfile.TemporaryDirectory(prefix='rag-answer-') as cwd:
            sdk_config = CodexConfig(codex_bin=binary, cwd=cwd, config_overrides=tuple(overrides))
            with Codex(sdk_config) as codex:
                thread = codex.thread_start(cwd=cwd, model=model, sandbox=Sandbox.read_only,
                                            approval_mode=ApprovalMode.deny_all, ephemeral=True,
                                            base_instructions='只根据用户提供的证据生成回答。禁止使用工具或读取外部内容。')
                result = thread.run(prompt, output_schema=schema)
                if str(result.status.value) != 'completed' or not result.final_response:
                    raise RuntimeError('Codex 未完成生成或返回空答案')
                item_types = [getattr(getattr(item,'root',item),'type','unknown') for item in result.items]
                unexpected = [kind for kind in item_types if kind not in ('userMessage','agentMessage','reasoning')]
                if unexpected:
                    raise RuntimeError(f'生成阶段发生额外工具活动，拒绝作为证据限定回答：{unexpected}')
                answer = json.loads(result.final_response)
                validate_answer(answer, valid_citations)
                return dict(answer, model=model or 'Codex 当前配置的默认模型',
                            item_types=item_types,
                            usage=result.usage.model_dump(mode='json') if result.usage else None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def validate_answer(answer, valid_citations):
    if answer.get('status') not in ('answered','insufficient','conflict') or not isinstance(answer.get('answer'),str):
        raise ValueError('Codex 返回格式无效')
    citations = answer.get('citations')
    if not isinstance(citations,list) or not all(isinstance(c,str) for c in citations):
        raise ValueError('Codex 引用格式无效')
    inline = set(re.findall(r'\[(E\d+)\]',answer['answer']))
    if not set(citations).issubset(valid_citations) or not inline.issubset(valid_citations):
        raise ValueError('Codex 引用了不存在的证据')
    if set(citations) != inline:
        raise ValueError('Codex 的正文引用与引用列表不一致')
    if answer['status'] in ('answered','conflict') and not inline:
        raise ValueError('Codex 的事实回答缺少来源引用')
