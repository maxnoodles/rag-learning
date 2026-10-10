#!/usr/bin/env python3
"""Export selected real user/final-answer pairs, never system/tool/reasoning text."""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('01a1122e-4072-7451-b959-1e3076be0744', '查找适合本机的轻量嵌入模型', 14),
    ('01a115f2-d742-7982-9161-182991fdf1e7', '理解合成数据与过拟合', 4),
    ('01a1167f-d3fe-7332-91f1-be593dc78911', '探索企业文化内化到模型的途径', 3),
    ('01a11687-a9a7-7b90-8222-3f3ceae7ddd7', '解释强化学习', 2),
    ('01a1194b-19e3-7530-a035-9a1c5dcce7f1', '设计不停机的索引切换', 2),
]


def extract(path):
    pairs, pending, previous = [], None, ''
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        event = json.loads(line)
        msg = event.get('payload', {})
        if event.get('type') != 'response_item' or msg.get('type') != 'message':
            continue
        text = ''.join(c.get('text', '') for c in msg.get('content', [])
                       if c.get('type') in ('input_text', 'output_text')).strip()
        if msg.get('role') == 'user':
            if not text or text.startswith(('<', '# Files mentioned')):
                continue
            pending = dict(question=text, previous_question=previous,
                           timestamp=event.get('timestamp'), user_line=line_no)
            previous = text
        elif msg.get('role') == 'assistant' and (
            msg.get('phase') == 'final_answer' or msg.get('channel') == 'final'
        ) and pending:
            # Source text is otherwise retained verbatim; machine citation footer is metadata.
            answer = re.sub(r'<oai-mem-citation>.*?</oai-mem-citation>', '', text,
                            flags=re.S).strip()
            pairs.append(dict(pending, answer=answer, assistant_line=line_no))
            pending = None
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=Path, default=Path.home()/'.codex/sessions')
    parser.add_argument('--output', type=Path, default=ROOT/'data/private/chat_samples')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for sid, title, count in SOURCES:
        paths = sorted(args.sessions.rglob(f'*{sid}*.jsonl'))
        choices = [extract(path) for path in paths]
        if not choices:
            raise RuntimeError(f'找不到聊天原文：{title}')
        pairs = max(choices, key=len)  # A fork may hold the completed conversation.
        pairs = [p for p in pairs if p['question'].strip() not in ('继续',)]
        if len(pairs) < count:
            raise RuntimeError(f'{title} 只有 {len(pairs)} 组可用问答，需要 {count} 组')
        selected = pairs[-count:] if sid.startswith('01a1194b') else pairs[:count]
        for pair in selected:
            n = len(records)+1
            timestamp = pair['timestamp']
            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).astimezone(
                ZoneInfo('Asia/Shanghai')).isoformat() if timestamp else '未知'
            metadata = dict(pair_id=f'chat-{n:03}', thread_id=sid, title=title,
                            timestamp=date, user_line=pair['user_line'],
                            assistant_line=pair['assistant_line'])
            prior = pair['previous_question']
            context = f'## 前一条用户问题（上下文）\n\n{prior}\n\n' if prior else ''
            body = f'{context}## 用户\n\n{pair["question"]}\n\n## 助手\n\n{pair["answer"]}\n'
            text = '---\n'+json.dumps(metadata, ensure_ascii=False)+'\n---\n\n'+body
            # Do not emit possible credentials into a corpus or terminal.
            if re.search(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})', text):
                raise RuntimeError(f'{metadata["pair_id"]} 检测到疑似凭据，未导出')
            target = args.output/f'{n:03}.md'
            target.write_text(text, encoding='utf-8')
            records.append(dict(metadata, filename=target.name, question=pair['question'],
                                sha256=hashlib.sha256(text.encode()).hexdigest()))
    # Delete only files previously managed by this exporter, never unrelated user documents.
    old = args.output/'corpus_manifest.json'
    if old.exists():
        for entry in json.loads(old.read_text()).get('records', []):
            filename = entry['filename']
            if filename not in {r['filename'] for r in records} and re.fullmatch(r'\d{3}\.md', filename):
                (args.output/filename).unlink(missing_ok=True)
    old.write_text(json.dumps({'source':'Codex original session messages',
                              'excluded':'system, developer, tools, reasoning, commentary, citation footers',
                              'records':records}, ensure_ascii=False, indent=2))
    print(f'已导出 {len(records)} 组真实问答 → {args.output}')
    for record in records:
        print(f'{record["pair_id"]} | {record["title"]} | {record["question"][:70]}')


if __name__ == '__main__':
    main()
