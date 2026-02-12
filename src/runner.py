import os
import json
import sys
import argparse
from typing import List

import numpy as np
from tqdm import tqdm

from src.types import Document, Chunk, Query
from src.registry import PROCESSOR_REG, CHUNKER_REG, ENCODER_REG, EVALUATOR_REG
from src.processors import BaseProcessor
from src.chunkers import BaseChunker
from src.encoders import BaseEncoder
from src.evaluators import BaseEvaluator
from src.io import *
from src.io.sink import write_trec_file

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _is_gutenqa_proposition_run(dataset_name: str, chunk_run_id: str) -> bool:
    """
    Proposition runs on GutenQA reuse paragraph chunk_ids.
    Their ranking should therefore be evaluated against the original
    paragraph chunks rather than the proposition file itself.
    """
    return dataset_name == 'GutenQA' and 'proposition' in chunk_run_id.lower()


def cmd_chunk(args: argparse.Namespace):
    args_dict = vars(args)

    processor_keys = ['dataset_name', 'data_folder']
    p_kw = {k: args_dict[k] for k in processor_keys}

    chunker_keys = ['embedding_model_name', 'tokenizer_name', 'sample', 'gen_backbone',
                    'generative_model_name', 'batch_size', 'resume']
    c_kw = {k: args_dict[k] for k in chunker_keys}

    chunker_kwargs_str = args_dict.get('chunker_kwargs') or "{}"
    try:
        extra_chunker_kwargs = json.loads(chunker_kwargs_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--chunker_kwargs must be valid JSON: {exc}") from exc

    if not isinstance(extra_chunker_kwargs, dict):
        raise ValueError("--chunker_kwargs must decode to a JSON object")

    c_kw.update(extra_chunker_kwargs)

    # create run id and path
    p_all_params = {"processor_name": args.processor_name, **p_kw}
    c_all_params = {"chunker_name": args.chunker, **c_kw}

    # build chunk run id
    chunk_run_id = args_dict.get('chunk_run_id') or build_chunk_run_id(p_all_params, c_all_params)
    P = Paths(
        dataset_name=p_kw['dataset_name'],
        base_dir=args.output_folder
    )

    chunks_dir = P.cs_dir(chunk_run_id)
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    else:
        if not args.resume:
            raise ValueError(f"{chunks_dir} already exists! Please remove it and try again.")
        print(f"[chunk] Resuming existing run at {chunks_dir}")
    chunks_output_path = P.cs_chunks_path(chunk_run_id)

    # run processor and chunker
    processor: BaseProcessor = PROCESSOR_REG.get(p_all_params['processor_name'])(**p_kw)

    c_kw['chunk_sink_path'] = chunks_output_path
    c_kw['resume'] = args.resume

    if c_all_params['chunker_name'] == "LumberChunker":
        if p_kw['dataset_name'] == "GutenQA":
            c_kw['granularity'] = "paragraph"
        else:
            c_kw['granularity'] = "sentence"

    print(c_kw)
    chunker: BaseChunker = CHUNKER_REG.get(c_all_params['chunker_name'])(**c_kw)

    # Special case: For GutenQA + Proposition, load existing chunks instead of docs
    if p_kw['dataset_name'] == 'GutenQA' and c_all_params['chunker_name'] == "Proposition":
        # Load pre-existing paragraph chunks to split into propositions
        paragraph_chunk_path = f"{args.output_folder}/GutenQA/chunks/ParagraphChunker/chunks.jsonl"
        chunks = load_chunks(paragraph_chunk_path)
        # Pass paragraph chunks to proposition chunker
        print(f"Loading {len(chunks)} paragraph chunks for proposition splitting")
        chunker.chunk(raw_docs=chunks)
    else:
        # Normal flow: load docs and chunk them
        docs: List[Document] = processor.load_corpus()
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
    args_dict = vars(args)

    # build embd_run_id run id
    encoder_keys = ['backbone', 'model_name', 'batch_size']
    raw_e_kw = {k: args_dict[k] for k in encoder_keys}
    e_all_params = {"encoder_name": args_dict['encoder_name'], **raw_e_kw}

    embd_run_id = build_emb_run_id(chunk_run_id=args.chunk_run_id, encoder=e_all_params)

    P = Paths(dataset_name=args_dict['dataset_name'], base_dir=args_dict['output_folder'])

    call_kwargs = {
        "batch_size": raw_e_kw['batch_size'],
    }

    # If --query is NOT specified, encode passages.
    if not args.query:
        embeddings_dir = P.er_dir(args.chunk_run_id, embd_run_id)

        # CHECK FIRST before creating anything
        if os.path.exists(embeddings_dir):
            raise ValueError(f"{embeddings_dir} already exists! Please remove it and try again.")

        # Now create the directory
        os.makedirs(embeddings_dir)

        # Get the output path (after directory is created)
        embeddings_output_path = P.er_embeddings_pkl(args.chunk_run_id, embd_run_id)

        # Initialize encoder
        init_kwargs = {
            "backbone": raw_e_kw['backbone'],
            "embed_sink_path": embeddings_output_path,
            "backbone_kwargs": {
                "model_name": raw_e_kw['model_name']
            }
        }

        if init_kwargs['backbone'] in ['OpenAI']:
            if API_KEY is None:
                raise ValueError(f"Backbone {args_dict['backbone']} API key is required")
            init_kwargs['backbone_kwargs']['api_key'] = API_KEY

        encoder: BaseEncoder = ENCODER_REG.get(args_dict['encoder_name'])(**init_kwargs)

        chunk_path = P.cs_chunks_path(args.chunk_run_id)
        chunks = load_chunks(chunk_path)
        encoder.encode_passages(chunks=chunks, **call_kwargs)

        # write manifest
        if init_kwargs['backbone_kwargs'].get('api_key'):
            init_kwargs['backbone_kwargs'].pop('api_key')

        write_embedding_manifest(
            paths=P,
            chunk_run_id=args.chunk_run_id,
            embed_run_id=embd_run_id,
            encoder=init_kwargs | call_kwargs,
        )
        print(f'[embeddings] wrote -> {embeddings_output_path}')

    # If --query is specified, encode queries.
    if args.query is True:

        query_embed_run_id = build_query_embedding_run_id(qs_id=args.query_run_id, encoder=e_all_params)
        query_embedding_dir = P.q_embed_dir(args.query_run_id, query_embed_run_id)

        if os.path.exists(query_embedding_dir):
            print(f"[query embeddings] Skipping: {query_embedding_dir} already exists.")
        else:
            print(f"[query embeddings] Creating directory: {query_embedding_dir}")
            os.makedirs(query_embedding_dir)

            query_embeddings_output_path = P.q_embeddings_pkl(args.query_run_id, query_embed_run_id)

            init_kwargs = {
                "backbone": raw_e_kw['backbone'],
                "embed_sink_path": None,  # Not used for query encoding
                "backbone_kwargs": {
                    "model_name": raw_e_kw['model_name']
                }
            }

            if init_kwargs['backbone'] in ['openai']:
                if API_KEY is None:
                    raise ValueError(f"Backbone {args_dict['backbone']} API key is required")
                init_kwargs['backbone_kwargs']['api_key'] = API_KEY

            encoder: BaseEncoder = ENCODER_REG.get(args_dict['encoder_name'])(**init_kwargs)

            query_path = f"{args.output_folder}/{args.dataset_name}/queries/{args.query_run_id}/queries.jsonl"
            queries = load_queries(query_path)
            encoder.encode_queries(queries=queries,
                                   query_sink_path=query_embeddings_output_path,
                                   batch_size=raw_e_kw['batch_size'], )

            # write manifest
            if init_kwargs['backbone_kwargs'].get('api_key'):
                init_kwargs['backbone_kwargs'].pop('api_key')

            write_query_embedding_manifest(
                paths=P,
                query_run_id=args.query_run_id,
                q_embed_id=query_embed_run_id,
                encoder=init_kwargs | call_kwargs,
            )

            print(f'[query embeddings] wrote -> {query_embeddings_output_path}')


def cmd_evaluator(args: argparse.Namespace):
    import time

    start = time.perf_counter()

    # Validate skip-search arguments
    if args.skip_search and not args.trec_file:
        raise ValueError("--trec-file is required when --skip-search is used")

    if _is_gutenqa_proposition_run(args.dataset_name, args.chunk_run_id):
        # For proposition evaluation, load original paragraph chunks (before proposition splitting)
        chunk_path = f"{args.source_path}/{args.dataset_name}/chunks/ParagraphChunker/chunks.jsonl"
    else:
        chunk_path = f"{args.source_path}/{args.dataset_name}/chunks/{args.chunk_run_id}/chunks.jsonl"

    query_path = f"{args.source_path}/{args.dataset_name}/queries/{args.query_run_id}/queries.jsonl"

    # Load chunks and queries (always needed)
    print('Load chunks and queries')
    chunks = load_chunks(chunk_path)
    queries = load_queries(query_path)

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

    if args.skip_search:
        # Skip search mode: load existing TREC file
        print(f'Skip-search mode: Loading existing TREC file from {args.trec_file}')
        ranking_results = load_trec_file(args.trec_file)
        print('Loading time:', time.perf_counter() - start)

        mid = time.perf_counter()
        print("Load successfully!!!")

        # For GutenQA, convert dict to list of tuples format
        if evaluator_name == 'GutenQA':
            # Convert {query_id: {chunk_id: score}} to {query_id: [(chunk_id, score)]}
            ranking_results = {
                qid: sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                for qid, doc_scores in ranking_results.items()
            }

        # Build results dict manually for evaluation
        # We need to compute per-query metrics from the ranking_results
        qrels = {q.query_id: q.qrels for q in queries}

        if evaluator_name == 'beir':
            from beir.retrieval.evaluation import EvaluateRetrieval
            evaluator: BaseEvaluator = EVALUATOR_REG.get(evaluator_name)(
                scope=args.scope,
                similarity=args.similarity
            )
            retriever = EvaluateRetrieval()
            k_values = evaluator.k_values
            ndcg, _map, recall, precision = retriever.evaluate(qrels, ranking_results, k_values)

            # Calculate per-query scores
            per_query_eval = {}
            for q_id in tqdm(qrels.keys(), desc="Calculating Per-Query Metrics"):
                qrels_single = {q_id: qrels[q_id]}
                results_single = {q_id: ranking_results.get(q_id, {})}
                ndcg_q, _, recall_q, _ = retriever.evaluate(qrels_single, results_single, k_values)

                query_scores = {}
                if ndcg_q and recall_q:
                    for k in k_values:
                        query_scores[f"NDCG@{k}"] = ndcg_q.get(f"NDCG@{k}", 0.0)
                        query_scores[f"Recall@{k}"] = recall_q.get(f"Recall@{k}", 0.0)

                per_query_eval[q_id] = query_scores

            print(ndcg)
            print(recall)
            results = {'ndcg': ndcg, 'recall': recall, 'per_query_eval': per_query_eval,
                       'ranking_results': ranking_results}

        elif evaluator_name == 'GutenQA':
            from collections import defaultdict
            evaluator: BaseEvaluator = EVALUATOR_REG.get(evaluator_name)(
                scope=args.scope,
                similarity=args.similarity
            )
            k_values = evaluator.k_values

            # Compute GutenQA metrics from ranking_results
            chunk_id2text = {c.chunk_id: c.text for c in chunks}
            ranked_relevance_dict = {}

            from src.evaluators.qutenqa_evaluator import find_index_of_match, compute_DCG, compute_Recall

            for query in tqdm(queries, desc="Calculating GutenQA Metrics"):
                query_id = query.query_id
                re_chunk_list = [chunk_id2text.get(c_id) for c_id, _ in ranking_results.get(query_id, [])]
                relevance = find_index_of_match(re_chunk_list, query.chunk_must_Contain)
                ranked_relevance_dict[query_id] = relevance

            dcg_dict = defaultdict(list)
            recall_dict = defaultdict(list)

            for top_k in k_values:
                for _, relevance in ranked_relevance_dict.items():
                    dcg = compute_DCG(relevance[:top_k])
                    recall = compute_Recall(relevance[:top_k])
                    dcg_dict[top_k].append(dcg)
                    recall_dict[top_k].append(recall)

            per_query_eval = defaultdict(dict)
            for top_k in k_values:
                for i, (query_id, relevance) in enumerate(ranked_relevance_dict.items()):
                    dcg = compute_DCG(relevance[:top_k])
                    recall = compute_Recall(relevance[:top_k])
                    per_query_eval[query_id][f'DCG@{top_k}'] = dcg
                    per_query_eval[query_id][f'Recall@{top_k}'] = recall

            final_dcg_dict = {f'DCG@{k}': round(np.mean(v), 5) for k, v in dcg_dict.items()}
            final_recall_dict = {f'Recall@{k}': round(np.mean(v), 5) for k, v in recall_dict.items()}

            print(final_dcg_dict)
            print(final_recall_dict)

            results = {'dcg': final_dcg_dict, 'recall': final_recall_dict, 'per_query_eval': per_query_eval,
                       'ranking_results': ranking_results}

    else:
        # Normal mode: perform search
        chunk_embed_path = f"{args.source_path}/{args.dataset_name}/embeddings/{args.chunk_run_id}/{args.chunk_embedding_run_id}/embeddings.pkl"
        query_embed_embed_path = f"{args.source_path}/{args.dataset_name}/query_embeddings/{args.query_run_id}/{args.query_embedding_run_id}/embeddings.pkl"

        print('Load chunk embeddings and query embeddings')
        chunk_embs = load_pkl_embeddings(chunk_embed_path)
        query_embs = load_pkl_embeddings(query_embed_embed_path)
        print('Loading time:', time.perf_counter() - start)

        mid = time.perf_counter()
        print("Load successfully!!!")

        evaluator: BaseEvaluator = EVALUATOR_REG.get(evaluator_name)(
            scope=args.scope,
            similarity=args.similarity
        )

        results = evaluator.evaluate(queries=queries,
                                     query_embeddings=query_embs,
                                     chunks=chunks,
                                     chunk_embeddings=chunk_embs
                                     )

    # Create the output directory structure
    base_results_dir = os.path.join(args.source_path, args.dataset_name, 'results', args.chunk_run_id,
                                    args.chunk_embedding_run_id)
    os.makedirs(base_results_dir, exist_ok=True)

    # Save TREC file (only if not in skip-search mode)
    ranking_results = results.pop('ranking_results')
    if not args.skip_search:
        trec_file_path = os.path.join(base_results_dir, "result.trec")
        write_trec_file(trec_file_path, ranking_results, args.chunk_embedding_run_id, top_k=args.top_k)
        print(f'[TREC file] wrote -> {trec_file_path}')
    else:
        print(f'[TREC file] Skipped writing (using existing file: {args.trec_file})')

    # --- New Evaluation File Saving Logic ---
    per_query_eval = results.pop('per_query_eval')

    # Step 1: Collect all scores for each metric
    from collections import defaultdict
    metric_scores = defaultdict(list)
    for query_id, query_scores in per_query_eval.items():
        if query_scores is None:
            print(f"Warning: No scores found for query_id '{query_id}'. Skipping.")
            continue
        for metric_name, value in query_scores.items():
            cleaned_metric_name = metric_name.replace("NDCG@", "nDCG@")
            metric_scores[cleaned_metric_name].append((query_id, value))

    # Step 2: Write a separate file for each metric
    for metric_name, scores_list in metric_scores.items():
        # Create a valid filename
        filename = f"{metric_name}.eval"
        eval_file_path = os.path.join(base_results_dir, filename)

        total_score = 0
        with open(eval_file_path, 'w') as f:
            for query_id, value in scores_list:
                f.write(f"{query_id} {value}\n")
                total_score += value

        # Calculate and append the average
        if scores_list:
            average_score = total_score / len(scores_list)
            with open(eval_file_path, 'a') as f:
                f.write(f"average {average_score}\n")

        print(f'[evaluation] wrote -> {eval_file_path}')

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
    pc.add_argument("--gen_backbone")
    pc.add_argument("--generative_model_name")
    pc.add_argument("--batch_size", type=int)
    pc.add_argument("--chunker_kwargs", default="{}")
    pc.add_argument("--resume", action='store_true', help="Resume chunking into an existing run directory")
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
    peval.add_argument("--top_k", type=int, default=100, help="Number of top documents to save in TREC file.")
    peval.add_argument("--skip-search", action='store_true',
                       help="Skip search/ranking step and load existing TREC file for evaluation only.")
    peval.add_argument("--trec-file", type=str, help="Path to existing TREC file (required if --skip-search is used).")
    peval.set_defaults(func=cmd_evaluator)

    return parser


def main(argv: List[str]):
    parser = build_parser()
    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == '__main__':
    print(sys.argv[1:])
    main(sys.argv[1:])
