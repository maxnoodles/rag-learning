#!/usr/bin/env python3
"""一次性下载中英文重排模型；查询时只从本地加载。"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'artifacts/models/bge-reranker-base')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    target = snapshot_download('BAAI/bge-reranker-base', local_dir=str(args.output_dir),
                               allow_patterns=['*.json', '*.txt', 'sentencepiece.bpe.model',
                                               'model.safetensors', 'README.md'])
    print(f'重排模型已保存：{target}')


if __name__ == '__main__':
    main()
