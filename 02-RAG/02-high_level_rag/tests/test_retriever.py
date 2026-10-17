"""验证实体选择边界、BFS 层次与方向、完整关系及来源保留。"""
import copy
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from graph_rag.retriever import KnowledgeGraph, bfs_subgraph, retrieve, select_seed_entities


def fixture(chunk_id='chunk-1'):
    person = Node(id='张三', type='Person')
    project = Node(id='项目甲', type='Project')
    other = Node(id='项目乙', type='Project')
    tech = Node(id='Python', type='Technology')
    facts = [
        (person, 'LEADS', project, '张三负责项目甲。'),
        (person, 'PARTICIPATES_IN', project, '张三参与项目甲。'),
        (person, 'KNOWS', tech, '张三熟悉Python。'),
        (project, 'USES', tech, '项目甲使用Python。'),
        (project, 'DEPENDS_ON', other, '项目甲依赖项目乙。'),
        (other, 'DEPENDS_ON', project, '项目乙依赖项目甲。'),
    ]
    return GraphDocument(
        nodes=[person, project, other, tech],
        relationships=[Relationship(source=s, type=r, target=t, properties={'evidence': q})
                       for s, r, t, q in facts],
        source=Document(page_content=''.join(q for _, _, _, q in facts),
                        metadata={'chunk_id': chunk_id, 'source': 'fixture.txt'}),
    )


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph([fixture()])
        self.ref = next(ref for ref, key in self.graph.candidates().items() if key == ('Person', '张三'))

    @patch('graph_rag.retriever.run_codex')
    def test_select_existing_typed_entity(self, mock):
        mock.return_value = {'seed_keys': [self.ref], 'reason': '问题提及张三'}
        result = select_seed_entities('张三负责什么？', self.graph)
        self.assertEqual(result['entities'], [{'type': 'Person', 'id': '张三'}])
        schema = mock.call_args.args[1]
        self.assertEqual(schema['properties']['seed_keys']['items']['enum'], list(self.graph.candidates()))

    @patch('graph_rag.retriever.run_codex')
    def test_no_match_is_empty_subgraph(self, mock):
        mock.return_value = {'seed_keys': [], 'reason': '没有匹配实体'}
        result = retrieve('海王星公转多久？', self.graph)
        self.assertEqual(result['status'], 'no_match')
        self.assertEqual(result['subgraph']['nodes'], [])
        self.assertEqual(result['subgraph']['sources'], [])

    @patch('graph_rag.retriever.run_codex')
    def test_invalid_reference_and_excessive_seeds_rejected(self, mock):
        for refs in [['N999'], [1], list(self.graph.candidates())]:
            with self.subTest(refs=refs):
                mock.return_value = {'seed_keys': refs, 'reason': ''}
                with self.assertRaises(ValueError):
                    select_seed_entities('问题', self.graph, max_seeds=1)

    @patch('graph_rag.retriever.run_codex')
    def test_multiple_seeds_and_duplicates(self, mock):
        refs = list(self.graph.candidates())[:2]
        mock.return_value = {'seed_keys': [*refs, refs[0]], 'reason': '两个实体'}
        result = select_seed_entities('比较两个实体', self.graph)
        self.assertEqual(len(result['entities']), 2)

    @patch('graph_rag.retriever.run_codex')
    def test_empty_graph_and_invalid_parameters_skip_model(self, mock):
        self.assertEqual(select_seed_entities('问题', KnowledgeGraph([]))['entities'], [])
        for kwargs in [{'hops': -1}, {'direction': 'wrong'}, {'max_nodes': 0}]:
            with self.assertRaises(ValueError):
                retrieve('问题', self.graph, **kwargs)
        with self.assertRaises(ValueError):
            retrieve(' ', self.graph)
        mock.assert_not_called()


class BFSTests(unittest.TestCase):
    def setUp(self):
        self.graph = KnowledgeGraph([fixture()])

    def test_outgoing_layers_cycles_and_non_tree_edges(self):
        result = bfs_subgraph(self.graph, [('Person', '张三')], direction='out', hops=2)
        distances = {n['id']: n['distance'] for n in result['nodes']}
        self.assertEqual(distances, {'张三': 0, '项目甲': 1, 'Python': 1, '项目乙': 2})
        edges = {(r['source']['id'], r['type'], r['target']['id']) for r in result['relationships']}
        # Python 已通过 KNOWS 访问，USES 仍须保留；边界节点的回边也须保留。
        self.assertIn(('项目甲', 'USES', 'Python'), edges)
        self.assertIn(('项目乙', 'DEPENDS_ON', '项目甲'), edges)
        self.assertEqual(len(edges), 6)
        self.assertFalse(result['truncated'])

    def test_incoming_and_bidirectional_preserve_fact_direction(self):
        incoming = bfs_subgraph(self.graph, [('Technology', 'Python')], direction='in', hops=1)
        self.assertEqual({n['id'] for n in incoming['nodes']}, {'Python', '张三', '项目甲'})
        self.assertTrue(any(r['source']['id'] == '项目甲' and r['target']['id'] == 'Python'
                            for r in incoming['relationships']))
        outgoing = bfs_subgraph(self.graph, [('Technology', 'Python')], direction='out', hops=1)
        self.assertEqual(len(outgoing['nodes']), 1)
        both = bfs_subgraph(self.graph, [('Technology', 'Python')], direction='both', hops=1)
        self.assertEqual(both['nodes'], incoming['nodes'])

    def test_multi_source_uses_minimum_distance(self):
        result = bfs_subgraph(self.graph, [('Person', '张三'), ('Project', '项目乙')], direction='out', hops=1)
        self.assertEqual({n['id']: n['distance'] for n in result['nodes']},
                         {'张三': 0, '项目乙': 0, '项目甲': 1, 'Python': 1})

    def test_node_budget_is_explicitly_reported(self):
        result = bfs_subgraph(self.graph, [('Person', '张三')], max_nodes=2)
        self.assertEqual(len(result['nodes']), 2)
        self.assertTrue(result['truncated'])

    def test_zero_hops_and_isolated_nodes_keep_source(self):
        result = bfs_subgraph(self.graph, [('Person', '张三')], hops=0)
        self.assertEqual(len(result['nodes']), 1)
        self.assertEqual(result['relationships'], [])
        self.assertEqual(len(result['sources']), 1)
        doc = GraphDocument(nodes=[Node(id='李四', type='Person')], relationships=[],
                            source=Document(page_content='李四是员工。', metadata={'chunk_id': 'isolated'}))
        result = bfs_subgraph(KnowledgeGraph([doc]), [('Person', '李四')])
        self.assertEqual(result['sources'][0]['page_content'], '李四是员工。')

    def test_merge_triples_preserves_multiple_sources(self):
        graph = KnowledgeGraph([fixture(), fixture('chunk-2')])
        result = bfs_subgraph(graph, [('Person', '张三')])
        self.assertEqual(len(result['relationships']), 6)
        self.assertTrue(all(len(r['evidence']) == 2 for r in result['relationships']))
        self.assertEqual(len(result['sources']), 2)

    def test_same_name_different_types_remain_distinct(self):
        doc = GraphDocument(nodes=[Node(id='张三', type='Person'), Node(id='张三', type='Project')],
                            relationships=[], source=Document(page_content='张三', metadata={'chunk_id': 'typed'}))
        self.assertEqual(len(KnowledgeGraph([doc]).nodes), 2)

    def test_conflicting_source_ids_are_rejected(self):
        changed = copy.deepcopy(fixture())
        changed.source.page_content += '\n不同内容'
        with self.assertRaisesRegex(ValueError, 'chunk_id'):
            KnowledgeGraph([fixture(), changed])

    def test_invalid_start_direction_or_depth(self):
        for seeds, kwargs in [([('Person', '不存在')], {}),
                              ([('Person', '张三')], {'hops': -1}),
                              ([('Person', '张三')], {'direction': 'wrong'})]:
            with self.assertRaises(ValueError):
                bfs_subgraph(self.graph, seeds, **kwargs)


if __name__ == '__main__':
    unittest.main()
