"""测试可观测行为与边界，不把小样例成功外推为问答质量保证。"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rag.embedder import Embedder

from rag.chunker import make_chunks
from rag.codex_adapter import validate_answer
from rag.context_builder import build_context, token_count
from rag.document_parser import parse_documents
from rag.schemas import Chunk, Document, Hit
from rag.vector_store import fingerprint, load_current, save_snapshot
from rag.retriever import retrieve
from rag.lexical_index import BM25Index, tokenize

# 单元测试直接使用存储层，未经过 Embedder.__init__，也需限制两个线程池。
import faiss
import torch
from sentence_transformers import SentenceTransformer
torch.set_num_threads(1)
faiss.omp_set_num_threads(1)


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=True):
        return list(text)+([0,1] if add_special_tokens else [])


def one_chunk(doc, start=0, end=None, ordinal=0):
    end=len(doc.text) if end is None else end
    return Chunk(f'{doc.doc_id}-{start}-{end}',doc.doc_id,ordinal,start,end,doc.text[start:end],doc.text[start:end],end-start)


class ParserTests(unittest.TestCase):
    def test_txt_bom_crlf_and_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'a.txt').write_bytes(b'\xef\xbb\xbfhello\r\nworld')
            (root/'b.md').write_text('# 标题\n\n正文')
            docs=parse_documents(root)
            self.assertEqual(len(docs),2)
            self.assertEqual(docs[0].text,'hello\nworld')

    def test_empty_document_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'a.txt').write_text('  ')
            with self.assertRaisesRegex(ValueError,'空文档'):
                parse_documents(Path(tmp))

    def test_encoding_error_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'a.txt').write_bytes(b'\xff')
            with self.assertRaises(UnicodeDecodeError):
                parse_documents(Path(tmp))

    def test_no_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'a.pdf').write_bytes(b'%PDF')
            with self.assertRaisesRegex(ValueError,'没有 TXT'):
                parse_documents(Path(tmp))

    def test_real_samples_have_source_and_question(self):
        docs=parse_documents(Path(__file__).resolve().parents[1]/'data/private/chat_samples')
        self.assertEqual(len(docs),25)
        self.assertTrue(all(d.question and d.metadata.get('thread_id') for d in docs))


class ChunkTests(unittest.TestCase):
    def test_unicode_coverage_budget_question_and_overlap(self):
        doc=Document('a','a.md','测试','第一段中文🙂。\n\n第二段中文。'*30,'为什么？')
        chunks=make_chunks([doc],CharacterTokenizer(),100,12)
        self.assertGreater(len(chunks),1)
        covered=set()
        for c in chunks:
            self.assertEqual(c.text,doc.text[c.start:c.end])
            self.assertLessEqual(c.token_count,100)
            self.assertIn('本组问题：为什么？',c.embedding_text)
            covered.update(range(c.start,c.end))
        self.assertEqual(len(covered),len(doc.text))
        self.assertTrue(all(b.start<a.end for a,b in zip(chunks,chunks[1:])))

    def test_different_pairs_do_not_mix(self):
        docs=[Document('a','a.md','A','回答 A','问题 A'),Document('b','b.md','B','回答 B','问题 B')]
        chunks=make_chunks(docs,CharacterTokenizer(),100,0)
        self.assertEqual([c.doc_id for c in chunks],['a','b'])

    def test_oversized_question_is_error(self):
        with self.assertRaisesRegex(ValueError,'问题太长'):
            make_chunks([Document('a','a.txt','A','正文','问题'*80)],CharacterTokenizer(),64,0)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            make_chunks([],CharacterTokenizer(),64,64)

    def test_recursive_paragraphs_before_sentences(self):
        text = '第一段。'*5+'\n\n'+'第二段。'*5+'\n\n'+'第三段。'*5
        doc = Document('a','a.txt','A',text)
        chunks = make_chunks([doc],CharacterTokenizer(),64,0)
        self.assertEqual([c.text for c in chunks],
                         ['第一段。'*5+'\n\n', '第二段。'*5+'\n\n', '第三段。'*5])

    def test_repeated_text_whitespace_and_zero_overlap_spans(self):
        doc = Document('a','a.txt','A','  '+('重复🙂。\n\n'*30)+'尾部  ')
        chunks = make_chunks([doc],CharacterTokenizer(),64,0)
        self.assertEqual(''.join(c.text for c in chunks),doc.text)
        self.assertEqual(chunks[0].start,0)
        self.assertEqual(chunks[-1].end,len(doc.text))
        self.assertTrue(all(a.end==b.start for a,b in zip(chunks,chunks[1:])))

    def test_overlap_respects_token_budget_with_repeated_text(self):
        doc = Document('a','a.txt','A','重复🙂。'*90)
        chunks = make_chunks([doc],CharacterTokenizer(),64,8)
        for a,b in zip(chunks,chunks[1:]):
            self.assertGreater(b.end,a.end)
            self.assertGreaterEqual(b.start,a.start)
            self.assertLessEqual(len(CharacterTokenizer().encode(
                doc.text[b.start:a.end],add_special_tokens=False)),8)
        self.assertEqual(chunks[-1].end,len(doc.text))
        self.assertTrue(all(c.token_count <= 64 for c in chunks))


class BM25Tests(unittest.TestCase):
    def test_chinese_and_case_insensitive_code_terms(self):
        index = BM25Index(['苹果向量数据库 FAISS IndexFlatIP', '香蕉强化学习奖励', '橙子文档切分'])
        self.assertEqual(index.search('向量数据库 indexflatip',3)[0][0],0)
        self.assertIn('faiss', tokenize('FAISS'))

    def test_common_terms_positive_and_no_match_empty(self):
        index = BM25Index(['相同 common','相同 common'])
        self.assertEqual(len(index.search('common',10)),2)
        self.assertEqual(index.search('不存在的 qwerty',10),[])
        self.assertEqual(index.search('？！',10),[])
        self.assertEqual(BM25Index(['','！！']).search('test',2),[])


class StoreAndRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.docs=[Document(str(i),f'{i}.md',str(i),f'正文 {i}') for i in range(3)]
        self.chunks=[one_chunk(d) for d in self.docs]
        v=np.zeros((3,768),dtype='float32')
        v[0,:2]=[1,0]; v[1,:2]=[0.99,0.1]; v[2,:2]=[0.6,0.8]
        self.vectors=v/np.linalg.norm(v,axis=1,keepdims=True)
        self.snapshot=save_snapshot(self.root,'same',{},self.docs,self.chunks,self.vectors)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_identity_vectors_and_text(self):
        snap,reason=load_current(self.root,'same')
        self.assertIsNotNone(snap)
        self.assertEqual(snap.generation,self.snapshot.generation)
        np.testing.assert_allclose(snap.vectors,self.vectors)
        self.assertEqual(snap.chunks,self.chunks)

    def test_identity_change_causes_rebuild(self):
        snap,reason=load_current(self.root,'different')
        self.assertIsNone(snap)
        self.assertIn('变化',reason)

    def test_corrupt_cache_rejected(self):
        folder=self.root/'generations'/self.snapshot.generation
        (folder/'vectors.faiss').write_bytes(b'broken')
        snap,reason=load_current(self.root,'same')
        self.assertIsNone(snap)
        self.assertIn('校验失败',reason)

    def test_failed_publish_preserves_previous(self):
        previous=(self.root/'CURRENT').read_text()
        with patch('rag.vector_store.os.replace',side_effect=OSError('simulated publish error')):
            with self.assertRaises(OSError):
                save_snapshot(self.root,'new',{},self.docs,self.chunks,self.vectors)
        self.assertEqual((self.root/'CURRENT').read_text(),previous)
        self.assertIsNotNone(load_current(self.root,'same')[0])

    def test_mmr_relevance_and_diversity(self):
        q=self.vectors[0]
        recalled,selected=retrieve(self.snapshot,q,3,2,0.4,vector_weight=1)
        self.assertEqual(recalled[0].chunk.doc_id,'0')
        self.assertEqual([s.chunk.doc_id for s in selected],['0','2'])
        _,relevant=retrieve(self.snapshot,q,3,2,1,vector_weight=1)
        self.assertEqual([s.chunk.doc_id for s in relevant],['0','1'])

    def test_fetch_beyond_corpus_no_negative_ids(self):
        recalled,selected=retrieve(self.snapshot,self.vectors[0],20,4,0.7,query='正文')
        self.assertEqual(len(recalled),3)
        self.assertEqual(len(selected),3)

    def test_bm25_recalls_keyword_missing_from_vector_candidates(self):
        docs=[Document('a','a.md','A','alpha background'),
              Document('b','b.md','B','beta background'),
              Document('c','c.md','C','IndexFlatIP 归一化向量检索')]
        snap=save_snapshot(self.root,'hybrid',{},docs,[one_chunk(d) for d in docs],self.vectors)
        recalled,selected=retrieve(snap,self.vectors[0],1,1,1,query='indexflatip')
        self.assertEqual(selected[0].chunk.doc_id,'a')  # 同权重 RRF 同分时稳定按 ID 排序。
        recalled,selected=retrieve(snap,self.vectors[0],1,1,1,query='indexflatip',vector_weight=.4)
        self.assertEqual(selected[0].chunk.doc_id,'c')
        self.assertIsNone(selected[0].vector_rank)
        self.assertEqual(selected[0].bm25_rank,1)
        self.assertGreater(selected[0].bm25_score,0)
        self.assertAlmostEqual(selected[0].rrf_score,.6/61)
        self.assertAlmostEqual(selected[0].relevance,.6)

    def test_rrf_deduplicates_and_preserves_scores(self):
        recalled,selected=retrieve(self.snapshot,self.vectors[0],3,3,1,query='正文 0')
        self.assertEqual(len({h.chunk.chunk_id for h in recalled}),3)
        first=recalled[0]
        self.assertEqual(first.chunk.doc_id,'0')
        self.assertEqual(first.vector_rank,1)
        self.assertEqual(first.bm25_rank,1)
        self.assertAlmostEqual(first.rrf_score,1/61)
        self.assertAlmostEqual(first.relevance,1)
        self.assertIsNotNone(first.vector_score)
        self.assertIsNotNone(first.bm25_score)

    def test_pure_bm25_and_unmatched_query(self):
        recalled,selected=retrieve(self.snapshot,None,3,2,1,query='2',vector_weight=0)
        self.assertEqual([h.chunk.doc_id for h in recalled],['2'])
        self.assertEqual(selected[0].chunk.doc_id,'2')
        self.assertEqual(retrieve(self.snapshot,None,3,2,1,query='qwerty',vector_weight=0),([],[]))
        recalled,_=retrieve(self.snapshot,self.vectors[0],3,2,1,query='qwerty')
        self.assertEqual(recalled[0].chunk.doc_id,'0')
        self.assertAlmostEqual(recalled[0].relevance,1)

    def test_hybrid_validation(self):
        for kwargs in [dict(query=''), dict(query='正文',vector_weight=-1),
                       dict(query='正文',rrf_k=0)]:
            with self.assertRaises(ValueError):
                retrieve(self.snapshot,self.vectors[0],**kwargs)

    def test_bm25_restored_from_same_snapshot_and_removed_chunks_excluded(self):
        restored,_=load_current(self.root,'same')
        self.assertEqual(restored.bm25.search('2',10),self.snapshot.bm25.search('2',10))
        save_snapshot(self.root,'new',{},self.docs[:1],self.chunks[:1],self.vectors[:1])
        restored,_=load_current(self.root,'new')
        self.assertEqual(restored.bm25.search('2',10),[])

    def test_new_generation_replaces_removed_document(self):
        new=save_snapshot(self.root,'new',{},self.docs[:1],self.chunks[:1],self.vectors[:1])
        restored,_=load_current(self.root,'new')
        self.assertEqual(restored.generation,new.generation)
        self.assertEqual(list(restored.documents),['0'])
        self.assertEqual(len(self.snapshot.chunks),3)  # Already-open readers remain consistent.

    def test_fingerprint_tracks_text_model_and_chunk_settings(self):
        docs=self.root/'docs'; model=self.root/'model'; docs.mkdir(); model.mkdir()
        (docs/'a.txt').write_text('first'); (model/'model.safetensors').write_bytes(b'fake-test-weight')
        a,_=fingerprint(docs,model,512,64)
        (docs/'a.txt').write_text('second'); b,_=fingerprint(docs,model,512,64)
        c,_=fingerprint(docs,model,256,32)
        (model/'model.safetensors').write_bytes(b'changed-test-weight'); d,_=fingerprint(docs,model,256,32)
        self.assertEqual(len({a,b,c,d}),4)


class ContextTests(unittest.TestCase):
    # Separate context tests use the same small normalized-vector fixture.
    def test_duplicate_hit_and_full_pair_completion(self):
        doc=Document('a','a.md','A','头部问题\n答案第一部分\n答案第二部分','问题')
        chunks=[one_chunk(doc,0,10),one_chunk(doc,7,len(doc.text),1)]
        snap=type('SnapshotFixture',(),{'documents':{'a':doc}})()
        evidence,context,events=build_context(snap,[Hit(chunks[0],1),Hit(chunks[0],1),Hit(chunks[1],.9)])
        self.assertEqual(len(evidence),1)
        self.assertEqual(evidence[0].text,doc.text)
        self.assertEqual(len(evidence[0].chunk_ids),2)
        self.assertTrue(evidence[0].expanded)

    def test_budget_never_exceeded_and_disjoint_ranges_not_filled(self):
        doc=Document('a','a.md','A','左侧命中。'+'中间不相关内容。'*400+'右侧命中。')
        chunks=[one_chunk(doc,0,6),one_chunk(doc,len(doc.text)-6)]
        snap=type('SnapshotFixture',(),{'documents':{'a':doc}})()
        evidence,context,events=build_context(snap,[Hit(c,1) for c in chunks],128)
        self.assertLessEqual(token_count(context),128)
        self.assertNotIn('中间不相关内容',context)
        self.assertTrue(any('超预算' in e for e in events))

    def test_identical_text_keeps_both_sources(self):
        docs=[Document('a','a.md','A','相同答案'),Document('b','b.md','B','相同答案')]
        snap=type('SnapshotFixture',(),{'documents':{d.doc_id:d for d in docs}})()
        evidence,_,_=build_context(snap,[Hit(one_chunk(d),1) for d in docs])
        self.assertEqual(len(evidence),1)
        self.assertEqual(len(evidence[0].sources),2)

    def test_real_chat_metadata_does_not_conflict_with_source_fields(self):
        doc=Document('a','a.md','标题','真实问答','问题',{'title':'标题','thread_id':'source'})
        snap=type('SnapshotFixture',(),{'documents':{'a':doc}})()
        evidence,_,_=build_context(snap,[Hit(one_chunk(doc),1)])
        self.assertEqual(evidence[0].sources[0]['thread_id'],'source')


class AnswerTests(unittest.TestCase):
    def test_valid_citation_and_refusal(self):
        validate_answer({'status':'answered','answer':'答案 [E1]','citations':['E1']},{'E1'})
        validate_answer({'status':'insufficient','answer':'证据不足','citations':[]},{'E1'})

    def test_fabricated_or_missing_citations_rejected(self):
        for result in [{'status':'answered','answer':'答案 [E9]','citations':['E9']},
                       {'status':'answered','answer':'答案','citations':[]},
                       {'status':'answered','answer':'答案 [E1]','citations':[]}]:
            with self.assertRaises(ValueError):
                validate_answer(result,{'E1'})


class EmbeddingGuardTests(unittest.TestCase):
    def fake_embedder(self,value):
        embedder=Embedder.__new__(Embedder)
        embedder.model=type('ModelFixture',(),{'encode':lambda self,*args,**kwargs:value})()
        return embedder

    def test_wrong_dimension_cannot_silently_reshape(self):
        with self.assertRaisesRegex(ValueError,'形状错误'):
            self.fake_embedder(np.ones((1,1536)))._encode(['query'])

    def test_nan_and_zero_vectors_rejected(self):
        for value in [np.zeros((1,768)),np.full((1,768),np.nan)]:
            with self.assertRaises(ValueError):
                self.fake_embedder(value)._encode(['query'])

    def test_output_is_float32_and_normalized(self):
        vectors=self.fake_embedder(np.ones((2,768)))._encode(['a','b'])
        self.assertEqual(vectors.dtype,np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)


if __name__=='__main__':
    unittest.main()
