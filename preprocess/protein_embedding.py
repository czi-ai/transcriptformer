import argparse
import gzip
import json
import logging
import os
import pickle
import shutil
import urllib.request
from pathlib import Path

import esm
import h5py
import numpy as np
import torch
from Bio import SeqIO
from esm import FastaBatchedDataset
from esm.data import Alphabet

STABLE_ID_DIR = "gene_protein_stable_ids/"
FASTA_MANIFEST = "fasta_manifest_pep.json"


def save_as_hdf5(data_dict, output_path):
    """Save dictionary as HDF5 file."""
    with h5py.File(output_path, "w") as f:
        # Store the keys as a dataset
        keys = list(data_dict.keys())
        f.create_dataset("keys", data=np.array(keys, dtype="S"))

        # Create a group for the arrays
        arrays_group = f.create_group("arrays")
        for key, value in data_dict.items():
            arrays_group.create_dataset(str(key), data=value)


def clean_sequence(seq: str):
    """
    Cleans the input protein sequence by replacing any asterisk (*) characters with the <unk> token.

    Args:
        seq (str): The input protein sequence.

    Returns
    -------
        str: The cleaned protein sequence with asterisks replaced by <unk>.
    """
    return seq.replace("*", "<unk>")


def pad_batch(toks, num_gpus):
    """
    Pads the batch to ensure its size is a multiple of the number of GPUs.

    Args:
        toks (torch.Tensor): The tokenized sequences.
        num_gpus (int): The number of GPUs.

    Returns
    -------
        torch.Tensor: The padded tokenized sequences.
    """
    batch_size = toks.size(0)
    if batch_size % num_gpus != 0:
        padding_size = num_gpus - (batch_size % num_gpus)
        padding = torch.zeros((padding_size, toks.size(1)), dtype=toks.dtype)
        toks = torch.cat([toks, padding], dim=0)
    return toks


def generate_embeddings(
    model: torch.nn.Module,
    alphabet: Alphabet,
    fasta: str,
    save_file: str,
    seq_length=1022,
    batch_size=16,
):
    """
    Generates embeddings for protein sequences from a given FASTA file using a pre-trained model.

    Args:
        model (torch.nn.Module): The pre-trained PyTorch model to use for generating embeddings.
        alphabet (Alphabet): The alphabet object used for encoding sequences.
        fasta (str): Path to the input FASTA file containing protein sequences.
        save_file (str): Path to save the generated embeddings.
        seq_length (int, optional): Maximum sequence length for the embeddings. Defaults to 1022.
        batch_size (int, optional): Batch size for processing. Defaults to 16.

    Returns
    -------
        None
    """
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = FastaBatchedDataset.from_file(fasta)

    num_tokens = 4096 * batch_size
    batches = dataset.get_batch_indices(num_tokens, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
        num_workers=0,
    )

    dataset.sequence_strs = [clean_sequence(seq) for seq in dataset.sequence_strs]

    if os.path.exists(save_file):
        os.remove(save_file)

    embeddings = {}
    num_gpus = torch.cuda.device_count()
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1} of {len(batches)}")
            if torch.cuda.is_available():
                toks = pad_batch(toks, num_gpus).to(device="cuda", non_blocking=True)

                out = model(toks, repr_layers=[33], return_contacts=False)

                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

                for i, label in enumerate(labels):
                    truncate_len = min(seq_length, len(strs[i]))
                    embedding = representations[33][i, 1 : truncate_len + 1].mean(0).numpy()

                    entry_id = label.split()[0]

                    if entry_id in embeddings:
                        embeddings[entry_id].append(embedding)
                    else:
                        embeddings[entry_id] = [embedding]

                # Dump as we go just in case pipeline crashes
                temp_save_file = save_file + ".tmp"
                pickle.dump(embeddings, open(temp_save_file, "wb"))

    averaged_embeddings = {k: np.mean(v, axis=0) for k, v in embeddings.items()}

    save_as_hdf5(averaged_embeddings, save_file)


def main():
    parser = argparse.ArgumentParser(description="Generate protein embeddings and convert to gene embeddings.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        required=False,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--organism_key",
        type=str,
        default="homo_sapiens",
        help="The organism key to generate protein embeddings for",
    )
    parser.add_argument(
        "--use_large_model",
        action="store_true",
        help="Whether to use the large ESM-2 model",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load FASTA URL manifest
    with open(FASTA_MANIFEST) as f:
        fasta_urls = json.load(f)

    if args.organism_key not in fasta_urls:
        raise ValueError(f"Organism {args.organism_key} is not a valid organism in the fasta manifest")

    # Create stable_id_dir if it doesn't exist
    stable_id_dir = Path(STABLE_ID_DIR)
    stable_id_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ESM-2 model
    if args.use_large_model:
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        suffix = "_large"
    else:
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        suffix = ""

    model.eval()  # disables dropout for deterministic results
    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    organism = args.organism_key
    fasta_url = fasta_urls[organism]["fa"]

    fasta_file = stable_id_dir / f"{organism}.fa"
    if not fasta_file.exists():
        logging.info(f"Downloading FASTA for {organism}")
        # Download and decompress in one step using Python
        with urllib.request.urlopen(fasta_url) as response:
            if response.headers.get("Content-Encoding") == "gzip":
                with gzip.GzipFile(fileobj=response) as gz_file:
                    with open(fasta_file, "w") as out_file:
                        shutil.copyfileobj(gz_file, out_file)
            else:
                with open(fasta_file, "w") as out_file:
                    shutil.copyfileobj(response, out_file)

    # Convert to gene IDs
    new_records = []
    seen_names = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        if not args.use_large_model:
            gene_id = record.description.split("gene:")[-1].split(" ")[0].strip().split(".")[0]
        else:
            if "gene_symbol" not in record.description:
                gene_id = record.description.split("gene:")[-1].split(" ")[0].strip().split(".")[0]
            else:
                gene_id = record.description.split("gene_symbol:")[-1].split(" ")[0].strip()

        if gene_id in seen_names:
            continue
        seen_names.add(gene_id)
        record.id = gene_id
        record.name = gene_id
        new_records.append(record)

    with open(fasta_file, "w") as f:
        SeqIO.write(new_records, f, "fasta")

    emb_file_name = output_dir / f"{organism}_gene{suffix}.h5"
    logging.info(f"Processing {fasta_file} for {organism}")

    generate_embeddings(
        model,
        alphabet,
        str(fasta_file),
        save_file=str(emb_file_name),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
