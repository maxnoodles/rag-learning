"""不调用模型，验证真实 LangChain 转换、来源和引文校验。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from graph_rag.pipeline import (build_graph_documents, create_transformer,
                                load_graph_documents, split_documents,
                                validate_graph_document)


def extraction():
    return {
        'nodes': [{'id': '张三', 'type': 'Person'}, {'id': '知识助手项目', 'type': 'Project'}],
        'relationships': [{
            'source_node_id': '张三', 'source_node_type': 'Person',
            'target_node_id': '知识助手项目', 'target_node_type': 'Project', 'type': 'LEADS',
            'properties': [{'key': 'evidence', 'value': '张三负责知识助手项目。'}],
        }],
    }


class GraphBuildTests(unittest.TestCase):
    def setUp(self):
        self.chunk = split_documents([Document(
            page_content='张三负责知识助手项目。', metadata={'document_id': 'team.txt', 'source': 'team.txt'},
        )])[0]

    def test_chinese_token_limit_and_stable_source(self):
        document = Document(page_content='张三负责知识助手项目。\n\n' * 40,
                            metadata={'document_id': 'long.txt', 'source': 'long.txt'})
        chunks = split_documents([document], chunk_size=40, chunk_overlap=8)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < c.metadata['token_count'] <= 40 for c in chunks))
        self.assertTrue(all('\ufffd' not in c.page_content for c in chunks))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertEqual([c.id for c in chunks],
                         [c.id for c in split_documents([document], 40, 8)])
        self.assertTrue(all(c.metadata['source'] == 'long.txt' for c in chunks))

    @patch('graph_rag.codex_model.run_codex', return_value=None)
    def test_real_transformer_roundtrip_and_strict_schema(self, mock):
        mock.return_value = extraction()
        transformer = create_transformer()
        with tempfile.TemporaryDirectory() as directory:
            graphs = build_graph_documents([self.chunk], transformer, directory)
            loaded = load_graph_documents(Path(directory) / 'graph_documents.json')
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].model_dump(), graphs[0].model_dump())
        self.assertEqual(loaded[0].source.metadata, self.chunk.metadata)
        self.assertEqual(loaded[0].relationships[0].properties['evidence'], self.chunk.page_content)
        schema = mock.call_args.args[1]
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(set(schema['required']), set(schema['properties']))

    @patch('graph_rag.codex_model.run_codex')
    def test_reject_invented_evidence_and_dangling_endpoint(self, mock):
        mock.return_value = extraction()
        graph = create_transformer().convert_to_graph_documents([self.chunk])[0]
        graph.relationships[0].properties['evidence'] = '张三使用Python。'
        with self.assertRaisesRegex(ValueError, '引文'):
            validate_graph_document(graph)
        graph.relationships[0].properties['evidence'] = self.chunk.page_content
        graph.nodes.pop()
        with self.assertRaisesRegex(ValueError, '端点'):
            validate_graph_document(graph)

    @patch('graph_rag.codex_model.run_codex')
    def test_isolated_entity_is_preserved(self, mock):
        mock.return_value = {'nodes': [{'id': '张三', 'type': 'Person'}], 'relationships': []}
        graph = create_transformer().convert_to_graph_documents([self.chunk])[0]
        validate_graph_document(graph)
        self.assertEqual(graph.nodes[0].id, '张三')
        self.assertEqual(graph.relationships, [])

    @patch('graph_rag.codex_model.run_codex')
    def test_schema_filters_wrong_relation_direction(self, mock):
        data = extraction()
        rel = data['relationships'][0]
        rel.update(source_node_id='知识助手项目', source_node_type='Project',
                   target_node_id='张三', target_node_type='Person')
        mock.return_value = data
        graph = create_transformer().convert_to_graph_documents([self.chunk])[0]
        self.assertEqual(graph.relationships, [])


if __name__ == '__main__':
    unittest.main()
