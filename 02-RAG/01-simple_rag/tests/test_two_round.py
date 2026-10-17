"""验证两轮合并、MMR 压缩后按原问题重排，以及失败报告可追溯。"""
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
from rag.retriever import merge_rounds, select_mmr
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

    def test_merge_keeps_strongest_round_scores_and_all_provenance(self):
        first, second = hit('a'), hit('a')
        first.relevance, first.rrf_score = .6, .01
        second.relevance, second.rrf_score = .9, .015
        merged = merge_rounds(('original',[first,hit('b')]),('abstract',[second]))
        self.assertEqual([h.chunk.chunk_id for h in merged],['a','b'])
        self.assertEqual(merged[0].relevance,.9)
        self.assertEqual(merged[0].rrf_score,.015)
        self.assertEqual(merged[0].retrieval_rounds,['original','abstract'])
        self.assertEqual(first.relevance,.6)
        self.assertEqual(second.retrieval_rounds,[])

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


class RetrievalMMRTests(unittest.TestCase):
    def candidates(self, scores=(1,.99,.98,0)):
        hits = [hit(name) for name in ['a','b','c','d']]
        for item, score in zip(hits,scores):
            item.relevance = score
        # 快照顺序刻意不同于重排顺序，防止把候选排名误当成向量行号。
        snapshot = SimpleNamespace(chunks=[hits[3].chunk,hits[2].chunk,hits[0].chunk,hits[1].chunk],
                                   vectors=np.asarray([[0,0,1],[0,1,0],[1,0,0],[1,0,0]],dtype=np.float32))
        return snapshot, hits

    def test_redundant_runner_up_is_replaced_from_full_pool(self):
        snapshot, hits = self.candidates()
        selected = select_mmr(snapshot,hits,2,.7)
        self.assertEqual([h.chunk.chunk_id for h in selected],['a','c'])
        self.assertAlmostEqual(selected[1].mmr_relevance,.98)
        self.assertAlmostEqual(selected[1].mmr_score,.7*.98)
        self.assertEqual(selected[1].redundancy,0)
        self.assertIsNone(selected[1].cross_encoder_score)  # MMR 不依赖 Cross-Encoder。
        self.assertEqual(selected[1].relevance,.98)
        self.assertIsNone(hits[2].mmr_relevance)
        self.assertEqual(hits[2].mmr_score,0)

    def test_lambda_one_preserves_retrieval_order_and_zero_starts_with_best(self):
        snapshot, hits = self.candidates()
        self.assertEqual([h.chunk.chunk_id for h in select_mmr(snapshot,hits,2,1)],['a','b'])
        self.assertEqual([h.chunk.chunk_id for h in select_mmr(snapshot,hits,2,0)],['a','c'])

    def test_ties_singleton_empty_and_more_than_available(self):
        snapshot, hits = self.candidates((.8,.8,.8,.8))
        selected = select_mmr(snapshot,hits,8)
        self.assertEqual([h.chunk.chunk_id for h in selected],['a','c','d','b'])
        self.assertTrue(all(h.mmr_relevance==.8 for h in selected))
        self.assertEqual(selected[-1].redundancy,1)
        self.assertEqual(len(select_mmr(snapshot,hits[:1],2)),1)
        self.assertEqual(select_mmr(snapshot,[],2),[])

    def test_negative_vector_relevance_and_rounding_are_clipped(self):
        snapshot, hits = self.candidates((1.0000001,.9,.8,-.1))
        selected = select_mmr(snapshot,hits,4)
        self.assertEqual(selected[0].mmr_relevance,1)
        self.assertEqual(selected[-1].mmr_relevance,0)
        self.assertEqual(hits[0].relevance,1.0000001)
        self.assertEqual(hits[-1].relevance,-.1)

    def test_negative_cosine_does_not_reward_irrelevant_candidate(self):
        snapshot, hits = self.candidates((1,.99,.98,.97))
        snapshot.vectors[0] = [-1,0,0]
        self.assertEqual([h.chunk.chunk_id for h in select_mmr(snapshot,hits,2)],['a','c'])

    def test_invalid_parameters_and_nonfinite_relevance_rejected(self):
        snapshot, hits = self.candidates()
        for n, weight in [(0,.7),(2,-.1),(2,1.1),(2,np.nan),(2,np.inf)]:
            with self.subTest(n=n,weight=weight), self.assertRaises(ValueError):
                select_mmr(snapshot,hits,n,weight)
        for score in [np.nan,np.inf,-np.inf]:
            hits[0].relevance = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                select_mmr(snapshot,hits)


class PipelineTests(unittest.TestCase):
    def invoke(self, folder, abstraction_error=None, vector_weight=.5, empty=False, reranker_error=None):
        a, b, c = hit('a'), hit('b'), hit('c')
        a.relevance, b.relevance, c.relevance = 1, .99, .98
        snapshot = SimpleNamespace(generation='test', chunks=[h.chunk for h in [a,b,c]],
                                   vectors=np.asarray([[1,0],[1,0],[0,1]],dtype=np.float32),
                                   documents={h.chunk.doc_id:Document(h.chunk.doc_id,h.chunk.doc_id+'.md',
                                              h.chunk.doc_id,h.chunk.text) for h in [a,b,c]})
        args = SimpleNamespace(query='原问题',output_json=folder/'report.json',docs=folder,
                               model_path=folder,index_dir=folder/'index',chunk_tokens=512,
                               overlap_tokens=64,rebuild=False,device='cpu',vector_weight=vector_weight,
                               rrf_k=60,fetch_k=10,top_k=4,mmr_k=2,top_n=2,mmr_lambda=.7,codex_model=None,
                               timeout=300,codex_bin=None,reranker_model_path=folder,
                               reranker_batch_size=4,context_tokens=6000,retrieve_only=True)
        events = []
        def search(*_args, **kwargs):
            events.append(kwargs['query'])
            if empty:
                return []
            return [a,b] if kwargs['query']=='原问题' else [b,c]
        def rewrite(*_args):
            events.append('abstraction')
            if abstraction_error:
                raise abstraction_error
            return {'abstract_query':'抽象问题'}
        # MMR 先从 a/b/c 中压缩为 a/c；Cross-Encoder 再把 c 排到 a 前面。
        reranker = self.fake_reranker = MergeAndRerankTests().fake_reranker([.1,3.5])
        reranker.device = 'cpu'
        if reranker_error:
            reranker.model.predict.side_effect = reranker_error
        with patch.object(run,'fingerprint',return_value=('id',{})), \
             patch.object(run,'load_current',return_value=(snapshot,'test cache')), \
             patch.object(run,'Embedder') as embedder, \
             patch.object(run,'recall',side_effect=search) as recall, \
             patch.object(run,'abstract_query',side_effect=rewrite), \
             patch.object(run,'Reranker',return_value=reranker), \
             patch.object(run,'generate') as generate, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if abstraction_error or reranker_error:
                with self.assertRaises(type(abstraction_error or reranker_error)):
                    run.execute(args)
            else:
                report = run.execute(args)
                self.assertEqual(report['status'],'completed')
                self.assertEqual([h['chunk']['chunk_id'] for h in report['mmr_candidates']],[] if empty else ['a','c'])
                self.assertEqual([h['chunk']['chunk_id'] for h in report['selected']],[] if empty else ['c','a'])
                self.assertEqual([h['chunk']['chunk_id'] for h in report['reranked']],[] if empty else ['c','a'])
                self.assertEqual(report['retrieval']['top_k'],4)
                self.assertEqual(report['retrieval']['mmr_k'],2)
                self.assertEqual(report['retrieval']['top_n'],2)
                self.assertEqual(report['retrieval']['mmr_lambda'],.7)
                self.assertEqual(report['retrieval']['method'],'two_round_rrf_mmr_cross_encoder')
                self.assertTrue(all(h['mmr_relevance'] is not None for h in report['selected']))
                self.assertEqual(recall.call_count,2)
                self.assertEqual([c.args[0] for c in embedder.return_value.query.call_args_list],
                                 ['原问题','抽象问题'] if vector_weight>0 else [])
                generate.assert_not_called()
                self.assertEqual(events,['原问题','abstraction','抽象问题'])
        return json.loads(args.output_json.read_text())

    def test_retrieve_only_still_rewrites_and_reranks_with_original_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp))
        self.assertIsNone(report['answer'])
        self.assertEqual(len(report['merged']),3)
        self.assertEqual(len(report['retrieval_rounds']),2)
        self.assertEqual(self.fake_reranker.model.predict.call_args.args[0],
                         [('原问题',h['chunk']['embedding_text']) for h in report['mmr_candidates']])
        self.assertTrue(all(h['cross_encoder_score'] is None for h in report['mmr_candidates']))

    def test_bm25_only_still_uses_stored_vectors_for_mmr(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp),vector_weight=0)
        self.assertEqual(len(report['selected']),2)

    def test_empty_candidates_skip_reranker_and_mmr(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp),empty=True)
        self.assertEqual(report['selected'],[])
        self.fake_reranker.model.predict.assert_not_called()

    def test_cli_rejects_invalid_mmr_lambda_before_execution(self):
        for value in ['-0.1','1.1','nan','inf']:
            with self.subTest(value=value), \
                 patch('sys.argv',['run.py','--mmr-lambda',value]), \
                 patch.object(run,'execute') as execute, \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as result:
                run.main()
            self.assertEqual(result.exception.code,2)
            execute.assert_not_called()

    def test_cli_defaults_and_invalid_candidate_counts(self):
        with patch('sys.argv',['run.py']), patch.object(run,'execute') as execute:
            self.assertEqual(run.main(),0)
        args = execute.call_args.args[0]
        self.assertEqual((args.fetch_k,args.top_k,args.mmr_k,args.top_n,args.mmr_lambda),
                         (10,4,4,2,.7))
        for options in [['--mmr-k','0'],['--mmr-k','9'],['--top-n','5'],
                        ['--top-k','11'],['--top-n','0']]:
            with self.subTest(options=options), \
                 patch('sys.argv',['run.py',*options]), \
                 patch.object(run,'execute') as execute, \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as result:
                run.main()
            self.assertEqual(result.exception.code,2)
            execute.assert_not_called()

    def test_reranker_failure_preserves_mmr_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp),reranker_error=RuntimeError('model failed'))
        self.assertEqual(report['status'],'failed')
        self.assertEqual(len(report['merged']),3)
        self.assertEqual([h['chunk']['chunk_id'] for h in report['mmr_candidates']],['a','c'])
        self.assertIn('model failed',report['error'])

    def test_abstraction_failure_preserves_first_round_without_falling_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.invoke(Path(tmp),TimeoutError('abstraction timed out'))
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['retrieval_rounds'][0]['name'],'original')
        self.assertEqual(len(report['retrieval_rounds']),1)
        self.assertIn('TimeoutError',report['error'])


if __name__ == '__main__':
    unittest.main()
