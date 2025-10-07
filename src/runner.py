import os
import json
import sys
import argparse
from typing import List

from src.types import Document, Chunk, Query
from src.registry import PROCESSOR_REG, CHUNKER_REG, ENCODER_REG, EVALUATOR_REG
from src.processors import BaseProcessor
from src.chunkers import BaseChunker
from src.encoders import BaseEncoder
from src.evaluators import BaseEvaluator
from src.io import *


API_KEY = None


def cmd_chunk(args: argparse.Namespace):

    args_dict = vars(args)

    processor_keys = ['dataset_name', 'data_folder']
    p_kw = {k:args_dict[k] for k in processor_keys}

    chunker_keys = ['embedding_model_name', 'tokenizer_name', 'sample']
    c_kw ={k:args_dict[k] for k in chunker_keys}

    # create run id and path
    p_all_params = {"processor_name": args.processor_name, **p_kw}
    c_all_params = {"chunker_name": args.chunker, **c_kw}

    # build chunk run id
    chunk_run_id = build_chunk_run_id(p_all_params, c_all_params)
    P = Paths(
        dataset_name=p_kw['dataset_name'],
        base_dir=args.output_folder
    )

    chunks_dir = P.cs_dir(chunk_run_id)
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    else:
        raise ValueError(f"{chunks_dir} already exists! Please remove it and try again.")
    chunks_output_path = P.cs_chunks_path(chunk_run_id)

    # run processor and chunker
    processor: BaseProcessor = PROCESSOR_REG.get(p_all_params['processor_name'])(**p_kw)
    docs: List[Document] = processor.load_corpus()

    c_kw['chunk_sink_path'] = chunks_output_path
    print(c_kw)
    chunker: BaseChunker = CHUNKER_REG.get(c_all_params['chunker_name'])(**c_kw)
    chunker.chunk(raw_docs=docs)

    write_chunk_manifest(
        paths=P,
        chunk_run_id=chunk_run_id,
        processor=p_all_params,
        chunker=c_all_params,

    )

    print(f'[chunk] wrote -> {chunks_output_path}')

    if args.query is True:

        # build query run id
        query_run_id = build_query_run_id(p_all_params)
        queries_dir = P.qs_dir(query_run_id)
        if not os.path.exists(queries_dir):
            os.makedirs(queries_dir)
            # print(f'{queries_dir} already exists! We will ')
        else:
            raise ValueError(f"{queries_dir} already exists! Please remove it and try again."
                             f"If you don't want to create query files, please delete '--query' in the command line.")
        queries_output_path = P.qs_queries_path(query_run_id)

        processor.load_query(sink_path=queries_output_path)

        write_query_manifest(
            paths=P,
            query_run_id=query_run_id,
            processor=p_all_params,
        )

        print(f'[Query] wrote -> {queries_output_path}')


def cmd_encoder(args: argparse.Namespace):

    # e_kw = _loads_json(args.encoder_kwargs)

    args_dict = vars(args)

    # build embd_run_id run id
    encoder_keys = ['backbone', 'model_name', 'batch_size']
    raw_e_kw = {k:args_dict[k] for k in encoder_keys}
    e_all_params = {"encoder_name": args_dict['encoder_name'], **raw_e_kw}

    embd_run_id = build_emb_run_id(chunk_run_id=args.chunk_run_id, encoder=e_all_params)

    P = Paths(dataset_name=args_dict['dataset_name'], base_dir=args_dict['output_folder'])
    embeddings_dir = P.er_dir(args.chunk_run_id, embd_run_id)


    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    else:
        raise ValueError(f"{embeddings_dir} already exists! Please remove it and try again."
                         f"If you don't want to create query embedding files, please delete '--query' in the command line.")

    # embeddings_output_path = P.er_embeddings_jsonl(args.chunk_run_id, embd_run_id)
    # embeddings_output_path = P.er_embeddings_gzip(args.chunk_run_id, embd_run_id)
    embeddings_output_path = P.er_embeddings_pkl(args.chunk_run_id, embd_run_id)
    #
    init_kwargs = {
        "backbone": raw_e_kw['backbone'],
        "embed_sink_path": embeddings_output_path,
        "backbone_kwargs": {
            "model_name": raw_e_kw['model_name']
        }
    }

    call_kwargs = {
        "batch_size": raw_e_kw['batch_size'],
    }

    if init_kwargs['backbone'] in ['openai']:
        if API_KEY is None:
            raise ValueError(f"Backbone {args_dict['backbone']} API key is required")
        init_kwargs['backbone_kwargs']['api_key'] = API_KEY

    chunk_path = P.cs_chunks_path(args.chunk_run_id)
    chunks = load_chunks(chunk_path)

    encoder: BaseEncoder = ENCODER_REG.get(args_dict['encoder_name'])(**init_kwargs)
    encoder.encode_passages(chunks=chunks, **call_kwargs)

    # write manifest
    if init_kwargs['backbone_kwargs'].get('api_key'):
        init_kwargs['backbone_kwargs'].pop('api_key')

    write_embedding_manifest(
        paths=P,
        chunk_run_id=args.chunk_run_id,
        embed_run_id=embd_run_id,
        encoder=init_kwargs|call_kwargs,
    )

    print(f'[embeddings] wrote -> {embeddings_output_path}')


    if args.query is True:

        query_embed_run_id = build_query_embedding_run_id(qs_id=args.query_run_id, encoder=e_all_params)

        query_embedding_dir = P.q_embed_dir(args.query_run_id, query_embed_run_id)
        if not os.path.exists(query_embedding_dir):
            os.makedirs(query_embedding_dir)
        else:
            raise ValueError(f"{query_embedding_dir} already exists! Please remove it and try again.")

        # query_embeddings_output_path = P.q_embeddings_jsonl(args.query_run_id, query_embed_run_id)
        query_embeddings_output_path = P.q_embeddings_pkl(args.query_run_id, query_embed_run_id)


        query_init_kwargs = {
            "backbone": raw_e_kw['backbone'],
            "query_embeddings_output_path": query_embeddings_output_path,
            "backbone_kwargs": {
            "model_name": raw_e_kw['model_name']
            }
        }

        query_path = f"{args.output_folder}/{args.dataset_name}/queries/{args.query_run_id}/queries.jsonl"
        queries = load_queries(query_path)
        encoder.encode_queries(queries=queries,
                               query_sink_path=query_embeddings_output_path,
                               batch_size=raw_e_kw['batch_size'],)

        write_query_embedding_manifest(
            paths=P,
            query_run_id=args.query_run_id,
            q_embed_id=query_embed_run_id,
            encoder=query_init_kwargs | call_kwargs,
        )

        print(f'[query embeddings] wrote -> {query_embeddings_output_path}')


def cmd_evaluator(args: argparse.Namespace):

    import time

    start = time.perf_counter()

    chunk_path = f"{args.source_path}/{args.dataset_name}/chunks/{args.chunk_run_id}/chunks.jsonl"
    query_path = f"{args.source_path}/{args.dataset_name}/queries/{args.query_run_id}/queries.jsonl"
    chunk_embed_path = f"{args.source_path}/{args.dataset_name}/embeddings/{args.chunk_run_id}/{args.chunk_embedding_run_id}/embeddings.pkl"
    query_embed_embed_path = f"{args.source_path}/{args.dataset_name}/query_embeddings/{args.query_run_id}/{args.query_embedding_run_id}/embeddings.pkl"

    # load chunks, queries, chunk embeddings and query embeddings
    print('Load chunks, queries, chunk embeddings and query embeddings')
    chunks = load_chunks(chunk_path)
    queries = load_queries(query_path)
    # chunk_embs = list(load_embeddings(chunk_embed_path))
    chunk_embs = load_pkl_embeddings(chunk_embed_path)

    # query_embs = load_queries_embeddings(query_embed_embed_path)
    query_embs = load_pkl_embeddings(query_embed_embed_path)
    print('Loading time:', time.perf_counter() - start)

    mid = time.perf_counter()

    print("Load successfully!!!")

    # register
    evaluator_mapping = {
        'GutenQA': 'GutenQA',
        'nfcorpus': 'beir',
        'arguana': 'beir',
        'fiqa': 'beir',
        'scidocs': 'beir',
        'scifact': 'beir',
        'trec-covid': 'beir',
    }

    evaluator_name = evaluator_mapping.get(args.dataset_name)
    if evaluator_name is None:
        raise ValueError(f'Evaluator {evaluator_mapping.get(args.dataset_name)} not found in Evaluator mapping.')

    evaluator: BaseEvaluator = EVALUATOR_REG.get(evaluator_name)(
        scope=args.scope,
        similarity=args.similarity
    )

    results = evaluator.evaluate(queries=queries,
                       query_embeddings=query_embs,
                       chunks=chunks,
                       chunk_embeddings=chunk_embs
                       )

    # save eval result
    save_path = 'src/results/result.jsonl'
    info = {'dataset': args.dataset_name,
            'chunker': args.chunk_run_id,
            'encoder': args.chunk_embedding_run_id.split('-')[0],
            'embedding_model': '-'.join(args.chunk_embedding_run_id.split('-')[1:]),}

    results = {**info, **results}
    write_evaluation_jsonl(save_path, results)
    print(f'[evaluation] wrote -> {save_path}')
    # print(results)

    print('Eval time:', time.perf_counter() - mid)

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog='runner', description='Modular paper reproduction CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_folder", default='src/output',
                        help="Output path for both chunk and embed")
    common.add_argument("--query", action='store_true')
    common.add_argument("--query_run_id", default=None)


    # chunk
    pc = sub.add_parser('chunker', parents=[common], help='Run Chunker only')
    pc.add_argument('--processor_name', required=True)
    pc.add_argument('--dataset_name', required=True)
    pc.add_argument('--data_folder', required=True)
    pc.add_argument('--sample', type=int)
    pc.add_argument("--chunker", required=True)
    pc.add_argument("--embedding_model_name")
    pc.add_argument("--tokenizer_name")
    pc.add_argument("--chunker_kwargs", default="{}")
    pc.set_defaults(func=cmd_chunk)


    # embed
    pe = sub.add_parser('encoder', parents=[common], help='Run Encoder only')
    pe.add_argument('--dataset_name', required=True)
    pe.add_argument('--chunk_run_id', required=True)
    pe.add_argument("--encoder_name", required=True)
    pe.add_argument("--backbone", required=True)
    pe.add_argument("--model_name", required=True)
    pe.add_argument("--batch_size", required=True, type=int)

    pe.set_defaults(func=cmd_encoder)


    # eval
    # We need to load
    #       - chunk files for text matching
    #       - query files, which contains 'contain must text'
    #       - chunk embeddings for similarity
    #       - query embeddings for similarity
    peval = sub.add_parser('evaluator', help='Run evaluators only')
    peval.add_argument('--chunk_run_id', required=True)
    peval.add_argument("--query_run_id", required=True)
    peval.add_argument("--chunk_embedding_run_id", required=True)
    peval.add_argument("--query_embedding_run_id", required=True)
    peval.add_argument("--dataset_name", required=True)
    peval.add_argument("--scope", choices=['document', 'corpus'], required=True,
                       help="Retrieve in a document or in all corpus")
    peval.add_argument("--similarity", choices=['cosine', 'dot'])
    peval.add_argument("--source_path", required=True)
    peval.add_argument("--output_folder", required=True)
    peval.set_defaults(func=cmd_evaluator)


    return parser


def main(argv: List[str]):

    parser = build_parser()
    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == '__main__':

    print(sys.argv[1:])
    main(sys.argv[1:])

)

    return args.func(args)


if __name__ == '__main__':

    print(sys.argv[1:])
    main(sys.argv[1:])

