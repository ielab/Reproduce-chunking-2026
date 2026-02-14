#!/usr/bin/env python3
"""
Generate LaTeX table comparing LumberChunker results on OpenAI embeddings (text-embedding-ada-002).
Compares original paper results with reproduced results on GutenQA dataset.
"""

import os
from pathlib import Path
from typing import Dict, Optional


def parse_eval_file(eval_path: str) -> Optional[float]:
    """
    Parse a DCG@10.eval file and extract the average score.
    """
    try:
        with open(eval_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('average'):
                    parts = line.split()
                    if len(parts) == 2:
                        return float(parts[1])
        return None
    except (FileNotFoundError, ValueError, IOError):
        return None


def get_chunker_display_name(chunker_name: str) -> str:
    """Convert chunker names to display names."""
    chunker_mapping = {
        'ParagraphChunker': 'Paragraph',
        'SentenceChunker': 'Sentence',
        'FixedSizeChunker-256': 'Fixed-size',
        'SemanticChunker': 'Semantic',
        'LumberChunker-Gemini': 'LumberChunker',
        'Proposition-Gemini': 'Proposition',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


def collect_results(
    base_path: str,
    chunkers: list,
    model: str = "text-embedding-ada-002",
    encoder: str = "RegularEncoder"
) -> Dict[str, float]:
    """
    Collect DCG@10 results for GutenQA dataset.
    """
    results = {}
    dataset = 'GutenQA'

    for chunker in chunkers:
        eval_path = os.path.join(
            base_path,
            dataset,
            'results',
            chunker,
            f'{encoder}-{model}',
            'DCG@10.eval'
        )

        score = parse_eval_file(eval_path)
        if score is not None:
            results[chunker] = score
        else:
            print(f"Warning: Could not read {eval_path}")

    return results


def generate_latex_table(
    results: Dict[str, float],
    chunkers: list,
    output_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table with three columns: Chunking Method, Original, Reproduced.
    """
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Comparison of LumberChunker results on OpenAI text-embedding-ada-002 (GutenQA, DCG@10)}')
    latex_lines.append(r'\label{tab:lumber_openai_comparison}')
    latex_lines.append(r'\begin{tabular}{lcc}')
    latex_lines.append(r'\toprule')
    latex_lines.append(r'Chunking Method & Original & Reproduced \\')
    latex_lines.append(r'\midrule')

    for chunker in chunkers:
        chunker_display = get_chunker_display_name(chunker)

        if chunker in results:
            reproduced = f'{results[chunker]:.4f}'
        else:
            reproduced = '---'

        # Original column left blank for user to fill from paper
        latex_lines.append(f'{chunker_display} &  & {reproduced} \\\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_table = '\n'.join(latex_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {output_path}")

    return latex_table


def main():
    """Main function to generate the results table."""

    # Configuration
    BASE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"

    # Define chunkers in order (same as other tables)
    CHUNKERS = [
        'ParagraphChunker',
        'SentenceChunker',
        'FixedSizeChunker-256',
        'SemanticChunker',
        'LumberChunker-Gemini',
        'Proposition-Gemini',
    ]

    MODEL = 'text-embedding-ada-002'
    ENCODER = 'RegularEncoder'

    OUTPUT_PATH = 'lumber_openai_comparison.tex'

    print("Collecting results...")
    results = collect_results(BASE_PATH, CHUNKERS, MODEL, ENCODER)

    print(f"Found {len(results)} result entries")

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(results, CHUNKERS, OUTPUT_PATH)

    print("\nGenerated LaTeX table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    print(f"\nDone! LaTeX table saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
