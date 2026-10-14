"""验证两轮候选不会提前丢失、重排面向原问题，以及失败报告可追溯。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import run
from rag.codex_adapter import abstract_query, validate_abstract_query
from rag.reranker import Reranker
from rag.retriever import merge_rounds
from rag.schemas import Chunk, Document, Hit


def hit(name):
    text = '片段 '+name
    return Hit(Chunk(name,name,0,0,len(text),text,'title: '+name+' | text: '+text,8),.8)


class MergeAndRerankTests(unittest.TestCase):
    def test_union_retains_second_round_and_provenance_without_mutating_inputs(self):
        a, b, c = hit('a'), hit('b'), hit('c')
        merged = merge_rounds(('original',[a,b]),('abstract',[b,c]))
        self.assertEqual([h.chunk.chunk_id for h in merged],['a','b','c'])
        self.assertEqual([h.retrieval_rounds for h in merged],
                         [['original'],['original','abstract'],['abstract']])
        self.assertEqual(b.retrieval_rounds,[])

    def fake_reranker(self, scores):
        model = Reranker.__new__(Reranker)
        model.batch_size = 4
        model.model = Mock()
        model.model.predict.return_value = np.asarray(scores)
        return model

    def test_second_round_only_candidate_can_win_and_all_candidates_are_scored(self):
        merged = merge_rounds(('original',[hit('a'),hit('b')]),('abstract',[hit('c')]))
        model = self.fake_reranker([.1,.2,3.5])
        ranked, selected = model.rerank('原问题',merged,2)
        self.assertEqual([h.chunk.chunk_id for h in selected],['c','b'])
        self.assertEqual(len(ranked),3)
        self.assertEqual(model.model.predict.call_args.args[0],
                         [('原问题',h.chunk.embedding_text) for h in merged])
        self.assertEqual(selected[0].cross_encoder_score,3.5)
        self.assertIsNone(merged[2].cross_encoder_score)

    def test_ties_empty_and_fewer_than_top_n(self):
        model = self.fake_reranker([1,1])
        _, selected = model.rerank('问题',[hit('a'),hit('b')],4)
        self.assertEqual([h.chunk.chunk_id for h in selected],['a','b'])
        self.assertEqual(model.rerank('问题',[],2),([],[]))
        self.assertEqual(model.model.predict.call_count,1)

    def test_invalid_scores_and_arguments_rejected(self):
        for scores in [[np.nan],[np.inf],[[1,2]],[]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.fake_reranker(scores).rerank('问题',[hit('a')])
        for query, n in [('',2),('问题',0)]:
            with self.assertRaises(ValueError):
                self.fake_reranker([1]).rerank(query,[hit('a')],n)


class AbstractionTests(unittest.TestCase):
    def test_invalid_output_and_blank_input_rejected(self):
        for value in [{},{'abstract_query':''},{'abstract_query':None},
                      {'abstract_query':'x'*4097},{'abstract_query':'a','answer':'b'},[]]:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                validate_abstract_query(value)
        with self.assertRaises(ValueError):
            abstract_query('  ')

    def test_uses_structured_codex_output_and_no_tools(self):
        result = SimpleNamespace(status=SimpleNamespace(value='completed'),
                                 final_response='{"abstract_query":" 通用问题 "}',
                                 items=[],usage=None)
        with patch('openai_codex.Codex') as codex:
            instance = codex.return_value.__enter__.return_value
            instance.thread_start.return_value.run.return_value = result
            output = abstract_query('具体问题')
            self.assertEqual(output['abstract_query'],'通用问题')
            thread_args = instance.thread_start.call_args.kwargs
            self.assertTrue(thread_args['ephemeral'])
            self.assertIn('禁止使用工具',thread_args['base_instructions'])
            self.assertIn('"具体问题"',instance.thread_start.return_value.run.call_args.args[0])

    def test_unexpected_tool_activity_is_rejected(self):
        result = SimpleNamespace(status=SimpleNamespace(value='completed'),
                                 final_response='{"abstract_query":"问题"}',
                                 items=[SimpleNamespace(type='commandExecution')],usage=None)
        with patch('openai_codex.Codex') as codex:
            codex.return_value.__enter__.return_value.thread_start.return_value.run.return_value = result
            with self.assertRaisesRegex(RuntimeError,'额外工具活动'):
                abstract_query('问题')


class PipelineTests(unittest.TestCase):
    def invoke(self, folder, abstraction_error=None):
        a, b, c = hit('a'), hit('b'), hit('c')
        snapshot = SimpleNamespace(generation='test', chunks=[h.chunk for h in [a,b,c]],
                                   documents={h.chunk.doc_id:Document(h.chunk.doc_id,h.chunk.doc_id+'.md',
                                              h.chunk.doc_id,h.chunk.text) for h in [a,b,c]})
        args = SimpleNamespace(query='原问题',output_json=folder/'report.json',docs=folder,
                               model_path=folder,index_dir=folder/'index',chunk_tokens=512,
                               overlap_tokens=64,rebuild=False,device='cpu',vector_weight=.5,
                               rrf_k=60,fetch_k=20,top_k=4,top_n=2,codex_model=None,
                               timeout=300,codex_bin=None,reranker_model_path=folder,
                               reranker_batch_size=4,context_tokens=6000,retrieve_only=True)
        events = []
        def search(*_args, **kwargs):
            events.append(kwargs['query'])
            return [a,b] if kwargs['query']=='原问题' else [b,c]
        def rewrite(*_args):
            events.append('abstraction')
            if abstraction_error:
                raise abstraction_error
            return {'abstract_query':'抽象问题'}
        reranker = self.fake_reranker = MergeAndRerankTests().fake_reranker([.1,.2,3.5])
        reranker.device = 'cpu'
        with patch.object(run,'fingerprint',return_value=('id',{})), \
             patch.object(run,'load_current',return_value=(snapshot,'test cache')), \
             patch.object(run,'Embedder') as embedder, \
             patch.object(run,'recall',side_effect=search) as recall, \
             patch.object(run,'abstract_query',side_effect=rewrite), \
             patch.object(run,'Reranker',return_value=reranker), \
             patch.object(run,'generate') as generate, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if abstraction_error:
                with self.assertRaises(type(abstraction_error)):
                    run.execute(args)
            else:
                report = run.execute(args)
                self.assertEqual(report['status'],'completed')
                self.assertEqual([h['chunk']['chunk_id'] for h in report['selected']],['c','b'])
                self.assertEqual(report['retrieval']['top_k'],4)
                self.assertEqual(report['retrieval']['top_n'],2)
                self.assertEqual(recall.call_count,2)
                self.assertEqual([c.args[0] for c in embedder.return_value.query.call_args_list],['原问题','抽象问题'])
                generate.assert_not_called()
                self.assertEqual(events,['原问题','abstraction','抽象问题'])
        return json.loads(args.output_json.read_text())

    def test_retrieve_only_still_rewrites_and_reranks_with_original_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp))
        self.assertIsNone(report['answer'])
        self.assertEqual(len(report['merged']),3)
        self.assertEqual(len(report['retrieval_rounds']),2)
        self.assertTrue(all(pair[0]=='原问题' for pair in self.fake_reranker.model.predict.call_args.args[0]))

    def test_abstraction_failure_preserves_first_round_without_falling_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp),TimeoutError('abstraction timed out'))
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['retrieval_rounds'][0]['name'],'original')
        self.assertEqual(len(report['retrieval_rounds']),1)
        self.assertIn('TimeoutError',report['error'])


if __name__ == '__main__':
    unittest.main()
