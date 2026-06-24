# Input data

Notebook inputs are persisted here. QM9 is stored in `QM9/`. For a real
DeepAffinity run, place the curated input at `deepaffinity.csv` with columns
`smiles`, `sps`, and `affinity`. The synthetic fallback is saved as
`deepaffinity_synthetic.csv` on first use.

GraphVAE stores its processed fixed-size QM9 tensors in `GraphVAE/`.

JT-VAE stores ZINC SMILES, its vocabulary, preprocessing shards, manifests, and
generated SMILES in `JunctionTreeVAE/`.

DeepDDS stores its frozen DrugComb table, balanced pairs, molecular graph cache,
split metadata, optional cell expression features, and predictions in `DeepDDS/`.

MolGAN stores fixed-size QM9 graph tensors and persistent RDKit QED/canonical
SMILES caches in `MolGAN/`.

HiGNN stores MoleculeNet downloads, BRICS fragment hierarchies, frozen scaffold
splits, and test predictions in `HiGNN/`.
