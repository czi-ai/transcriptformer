import os
import re
from itertools import product

import pandas as pd


def get_stable_id_mapping_from_gff3(
    gff3_file: str,
    organism_key: str,
    output_dir: str = "data/protein_embeddings/gene_protein_stable_ids",
):
    mapping_table = []
    lines = open(gff3_file).read().split("\n")

    start_index = None
    end_index = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "###":
            start_index = i + 1
            end_index = start_index
        else:
            i += 1
            continue

        while end_index < len(lines) and lines[end_index] != "###":
            end_index += 1

        data = lines[start_index:end_index]
        data = "\n".join(data)

        transcript_id_pattern = re.compile(r"transcript_id=([a-zA-Z0-9]+(?:\.[0-9]+)?)")
        protein_id_pattern = re.compile(r"protein_id=([a-zA-Z0-9]+(?:\.[0-9]+)?)")

        transcript_matches = transcript_id_pattern.findall(data)
        transcript_ids = transcript_matches if transcript_matches else []
        transcript_ids = list(set(transcript_ids))
        protein_matches = protein_id_pattern.findall(data)
        protein_ids = protein_matches if protein_matches else []
        protein_ids = list(set(protein_ids))

        gene_id_pattern = re.compile(r"gene_id=([a-zA-Z0-9]+)")
        gene_match = gene_id_pattern.search(data)
        gene_id = gene_match.group(1) if gene_match else None

        combinations = list(product([gene_id], transcript_ids, protein_ids))

        for gene_id, transcript_id, protein_id in combinations:
            mapping_table.append(
                {
                    "Protein stable ID version": protein_id,
                    "Gene stable ID": gene_id,
                    "Transcript stable ID": transcript_id,
                }
            )

        i = end_index

    mapping_table = pd.DataFrame(mapping_table)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    mapping_table.to_csv(f"{output_dir}/{organism_key}.tsv", sep="\t", index=False)


def get_stable_id_mapping_from_fasta(
    fasta_file: str,
    organism_key: str,
    output_dir: str = "data/protein_embeddings/gene_protein_stable_ids",
):
    mapping_table = []
    lines = open(fasta_file).read().split("\n")

    for line in lines:
        if line.startswith(">"):
            gene_symbol = line.split("gene_symbol:")[-1].split(" ")[0].strip()
            gene_id = line.split("gene:")[-1].split(" ")[0].strip()
            protein_id = line.split(" ")[0].split(">")[-1].strip()
            transcript_id = line.split("transcript:")[-1].split(" ")[0].strip()
            mapping_table.append(
                {
                    "Protein stable ID version": protein_id,
                    "Gene stable ID": gene_id,
                    "Transcript stable ID": transcript_id,
                    "Gene symbol": gene_symbol,
                }
            )

    mapping_table = pd.DataFrame(mapping_table)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    mapping_table.to_csv(f"{output_dir}/{organism_key}.tsv", sep="\t", index=False)
