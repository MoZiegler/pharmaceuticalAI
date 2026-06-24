from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


DEEP_LEARNING = notebook([
    md(r"""
# DeepLearning: the AI/PyTorch bridge for molecular ML

Role of this notebook: I am your graduate tutor in data science / computer science. You already program Python well and have beginner PyTorch/ML/transformer experience. The goal is not to make you memorize all of deep learning; it is to give you the exact conceptual and practical kit needed for the project notebooks:

- `MPNNs`: supervised graph neural networks for quantum chemistry.
- `DeepAffinity`: sequence + compound encoders for compound--protein affinity.
- `GraphVAE`, `JunctionTreeVariationalAutoencoder`, `MolGAN`: molecular generative models.
- `DeepDDS`, `HiGNN`: graph models, attention, drug synergy/property prediction.

Recommended external resources:

- [Dive into Deep Learning](https://d2l.ai/): open textbook with runnable PyTorch code. Most relevant: chapters 2--6, 10--13, 20.
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/): modern theory-first text. Use chapters 2--7 for foundations, 12 for transformers, 17--20 for generative models/graphs if present in your edition. Page numbers vary between HTML/PDF builds, so the chapter anchors are more reliable than printed pages.
- [fast.ai Practical Deep Learning for Coders](https://course.fast.ai/): excellent top-down practice, especially lessons 1--5.
- [Stanford CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/): graph ML/GNN lectures; especially message passing, GCN/GraphSAGE/GAT, and graph generation.
- [NYU Deep Learning](https://atcold.github.io/NYU-DLSP20/): great lectures on optimization, CNNs, sequence models, and representation learning.

Notebook style: each lesson has a mental model, formulas, a runnable mini-demo, and a connection to the molecular notebooks.
"""),
    code(r"""
from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
    HAS_PYG = True
except Exception as exc:
    HAS_PYG = False
    PYG_IMPORT_ERROR = repr(exc)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "Learning" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "Learning"
MODEL_DIR = PROJECT_ROOT / "models" / "Learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

print("PyTorch:", torch.__version__)
print("Device:", device)
print("PyG available:", HAS_PYG)
"""),
    md(r"""
## Lesson 1 — Tensors, shapes, and the habit of explicit dimensions

Deep learning code is mostly tensor algebra plus bookkeeping. The most important debugging skill is asking:

1. What does each axis mean?
2. Which operation mixes which axis?
3. Does batching change the meaning?

Typical molecular ML tensor meanings:

| object | common shape | meaning |
|---|---:|---|
| atom features | `[num_atoms, atom_dim]` | one row per atom |
| bond index | `[2, num_edges]` | source/target atom indices |
| protein tokens | `[batch, length]` | amino-acid token IDs |
| graph embedding | `[batch, hidden]` | one vector per molecule |
| pair embedding | `[batch, hidden]` | drug pair / compound-protein representation |
"""),
    code(r"""
x = torch.arange(24).reshape(2, 3, 4)
print("x shape:", x.shape)
print("batch 0:\n", x[0])
print("mean over sequence axis ->", x.float().mean(dim=1).shape)
print("mean over feature axis  ->", x.float().mean(dim=2).shape)
"""),
    md(r"""
## Lesson 2 — Supervised learning as empirical risk minimization

For data points $(x_i, y_i)$, a model $f_\theta$, and loss $\ell$, training minimizes

$$
\hat{R}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i), y_i).
$$

Regression notebooks use MSE/RMSE/MAE. Classification notebooks use cross-entropy/BCE and AUC. Generative notebooks optimize likelihood surrogates, ELBOs, or adversarial losses.

Resources:

- D2L chapters 3--4: linear regression and softmax regression.
- UDL chapters 2--5: supervised learning, losses, gradients, and training.
"""),
    code(r"""
# A tiny regression problem: learn y = 2x - 1 + noise.
n = 256
X = torch.linspace(-3, 3, n).unsqueeze(1)
y = 2 * X - 1 + 0.25 * torch.randn_like(X)

model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
X_d, y_d = X.to(device), y.to(device)

losses = []
for step in range(400):
    pred = model(X_d)
    loss = F.mse_loss(pred, y_d)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach().cpu()))

plt.plot(losses)
plt.yscale("log")
plt.title("Training loss")
plt.xlabel("step")
plt.ylabel("MSE")
plt.show()

with torch.no_grad():
    plt.scatter(X.numpy(), y.numpy(), s=8, alpha=0.35, label="data")
    plt.plot(X.numpy(), model(X_d).cpu().numpy(), color="black", label="MLP")
    plt.legend()
    plt.show()
"""),
    md(r"""
## Lesson 3 — Autograd and backpropagation

PyTorch records operations on tensors with `requires_grad=True`. Calling `loss.backward()` computes gradients by reverse-mode automatic differentiation.

The chain rule is the whole spell:

$$
\frac{\partial L}{\partial \theta}
=
\frac{\partial L}{\partial h}
\frac{\partial h}{\partial \theta}.
$$

Practical rules:

- Use `model.train()` while training, `model.eval()` while validating.
- Call `optimizer.zero_grad(set_to_none=True)` before backprop.
- Do not use `.item()` inside a differentiable expression.
- Use `with torch.no_grad()` for evaluation.
"""),
    code(r"""
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
pred = w * X[:5] + b
loss = ((pred - y[:5]) ** 2).mean()
loss.backward()
print("loss:", float(loss))
print("dL/dw:", w.grad.item(), "dL/db:", b.grad.item())
"""),
    md(r"""
## Lesson 4 — Optimization, regularization, and validation

The project notebooks repeatedly use:

- Adam/AdamW: adaptive optimizer, strong default.
- weight decay: penalizes large weights.
- dropout: stochastic feature removal.
- early stopping: stop when validation performance stops improving.
- scaffold splits: in chemistry, more realistic than random splits because similar molecules leak less easily.

For optimizer theory:

- D2L chapter 12: SGD, momentum, RMSProp, Adam, learning-rate schedules.
- NYU DL lectures on optimization.
"""),
    code(r"""
def train_val_split(n, frac=0.8):
    idx = torch.randperm(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]

train_idx, val_idx = train_val_split(len(X))
model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Dropout(0.05), nn.Linear(64, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

history = {"train": [], "val": []}
best_val = float("inf")
best_state = None

for epoch in range(300):
    model.train()
    pred = model(X_d[train_idx])
    loss = F.mse_loss(pred, y_d[train_idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val = F.mse_loss(model(X_d[val_idx]), y_d[val_idx])
    history["train"].append(float(loss.cpu()))
    history["val"].append(float(val.cpu()))
    if history["val"][-1] < best_val:
        best_val = history["val"][-1]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

torch.save({"model": best_state, "best_val_mse": best_val}, MODEL_DIR / "tiny_regression.pt")
plt.plot(history["train"], label="train")
plt.plot(history["val"], label="val")
plt.yscale("log")
plt.legend()
plt.title("Train/validation curves")
plt.show()
print("Saved:", MODEL_DIR / "tiny_regression.pt")
"""),
    md(r"""
## Lesson 5 — Embeddings: turning discrete symbols into vectors

Molecular notebooks often embed:

- atom type, degree, formal charge, chirality;
- bond type, conjugation, ring membership;
- amino-acid tokens for protein sequences;
- fragment/clique labels in junction-tree models.

An embedding table is just a learned lookup matrix $E \in \mathbb{R}^{V \times d}$.
"""),
    code(r"""
aa = "ACDEFGHIKLMNPQRSTVWY"
stoi = {ch: i + 1 for i, ch in enumerate(aa)}  # 0 reserved for padding
seqs = ["MKWVTFISLL", "ACDEFGHIK", "GGGGGG"]
max_len = max(map(len, seqs))
tokens = torch.zeros((len(seqs), max_len), dtype=torch.long)
for i, seq in enumerate(seqs):
    tokens[i, :len(seq)] = torch.tensor([stoi.get(ch, 0) for ch in seq])

emb = nn.Embedding(num_embeddings=len(stoi) + 1, embedding_dim=8, padding_idx=0)
z = emb(tokens)
mask = (tokens != 0).unsqueeze(-1)
pooled = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
print("tokens:", tokens.shape)
print("embedded:", z.shape)
print("mean pooled sequence embedding:", pooled.shape)
"""),
    md(r"""
## Lesson 6 — Attention and transformers, but only what you need here

Attention computes a weighted mixture of value vectors:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

In this repo:

- DeepAffinity uses sequence/compound representations and interpretable affinity signals.
- DeepDDS and HiGNN use attention-like mechanisms for drug-pair or feature-wise importance.
- Transformers are conceptually useful even when the model is a GNN: both pass contextual information between tokens/nodes.

Resources:

- D2L chapter 11: attention and transformers.
- UDL transformer chapter.
- Stanford CS224N transformer lectures if you want NLP depth.
"""),
    code(r"""
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = scores.softmax(dim=-1)
    return weights @ V, weights

torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out, weights = scaled_dot_product_attention(Q, K, V)
plt.imshow(weights[0].detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title("Toy attention matrix")
plt.xlabel("key position")
plt.ylabel("query position")
plt.show()
"""),
    md(r"""
## Lesson 7 — Graph neural networks and message passing

Most molecular graph models follow:

$$
m_v^{(t+1)} = \sum_{u \in \mathcal{N}(v)} M_t(h_v^{(t)}, h_u^{(t)}, e_{uv}),
$$

$$
h_v^{(t+1)} = U_t(h_v^{(t)}, m_v^{(t+1)}).
$$

Readout makes a graph-level vector:

$$
h_G = R(\{h_v^{(T)} : v \in G\}).
$$

Connections:

- MPNN paper: this framework was named and standardized for quantum chemistry.
- DeepDDS/HiGNN: GNNs plus attention and hierarchical structure.
- GraphVAE/MolGAN/JT-VAE: generate graphs rather than just predict labels.

Resources:

- Stanford CS224W graph neural network lectures.
- D2L has graph and recommender-adjacent material; CS224W is the better primary graph ML resource.
"""),
    code(r"""
if not HAS_PYG:
    print("PyTorch Geometric not available:", PYG_IMPORT_ERROR)
else:
    # Two tiny molecular-ish graphs, with categorical-ish node features replaced by random vectors.
    g1 = Data(
        x=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        y=torch.tensor([0.5]),
    )
    g2 = Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor([[0, 1, 2, 2, 3, 0], [1, 0, 1, 3, 2, 3]], dtype=torch.long),
        y=torch.tensor([1.0]),
    )
    batch = Batch.from_data_list([g1, g2]).to(device)
    conv = GCNConv(8, 16).to(device)
    head = nn.Linear(16, 1).to(device)
    h = conv(batch.x, batch.edge_index).relu()
    graph_h = global_mean_pool(h, batch.batch)
    pred = head(graph_h).view(-1)
    loss = F.mse_loss(pred, batch.y.float())
    loss.backward()
    print("node embeddings:", h.shape)
    print("graph embeddings:", graph_h.shape)
    print("loss:", float(loss.detach().cpu()))
"""),
    md(r"""
## Lesson 8 — Variational autoencoders: the generative-model skeleton

VAEs learn a latent distribution. Encoder:

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))).
$$

Decoder:

$$
p_\theta(x|z).
$$

Objective:

$$
\mathcal{L}_{ELBO}
=
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-
D_{KL}(q_\phi(z|x) \| p(z)).
$$

Connections:

- GraphVAE: decodes adjacency/node/edge tensors.
- JT-VAE: decodes a chemically constrained junction tree plus molecular graph.
"""),
    code(r"""
class TinyVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z)
        return recon, mu, logvar

theta = torch.linspace(0, 2 * math.pi, 512)
circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) + 0.05 * torch.randn(512, 2)
vae = TinyVAE().to(device)
opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
x = circle.to(device)
losses = []
for step in range(700):
    recon, mu, logvar = vae(x)
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + 0.05 * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach().cpu()))

with torch.no_grad():
    z = torch.randn(512, 2, device=device)
    samples = vae.dec(z).cpu()
plt.scatter(circle[:, 0], circle[:, 1], s=5, alpha=0.4, label="data")
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.4, label="VAE samples")
plt.axis("equal")
plt.legend()
plt.title("Tiny VAE intuition demo")
plt.show()
"""),
    md(r"""
## Lesson 9 — GANs and reinforcement signals

MolGAN is an implicit generative model:

- generator creates graph tensors;
- discriminator judges realism;
- reward network or RL term encourages molecular desirability.

Vanilla GAN objective:

$$
\min_G \max_D \mathbb{E}_{x\sim p_{data}}\log D(x)
+ \mathbb{E}_{z\sim p(z)}\log(1-D(G(z))).
$$

Practical moral: GAN training is less forgiving than supervised training. You monitor validity, uniqueness, novelty, mode collapse, and chemical constraints, not only loss curves.
"""),
    md(r"""
## Lesson 10 — How to read the project notebooks like a researcher

For each paper notebook, ask:

1. What is the input representation? SMILES, molecular graph, protein sequence, cell-line feature?
2. What inductive bias is built into the architecture? Message passing, hierarchy, junction tree, attention, adversarial learning?
3. What is the loss function?
4. What split/evaluation protocol prevents leakage?
5. What is saved to `data/` and `models/`, and how do I resume?
6. What would fail if the chemistry representation is wrong?

Suggested route through this repo:

1. Run `Environment_Hardware_Check.ipynb`.
2. Study this notebook through Lesson 7.
3. Run `Cheminformatics` Lessons 1--5.
4. Run `MPNNs/start_MPNN.ipynb`.
5. Then do `DeepDDS`/`HiGNN`.
6. Finally do the generative notebooks: `GraphVAE`, `JT-VAE`, `MolGAN`.
"""),
])


ADVANCED_CHEMISTRY = notebook([
    md(r"""
# AdvancedChemistry: chemistry theory for molecular deep learning

Role of this notebook: I am your tutor in organic, bio-, physical, and quantum chemistry. You have a university-minor chemistry background plus biochemistry/bioinformatics experience. The goal is to sharpen the chemistry needed to understand the paper notebooks, not to rebuild a full chemistry degree.

Relevant project notebooks:

- MPNNs: quantum-chemical molecular properties, atomization energies, dipoles, HOMO/LUMO, gaps.
- DeepAffinity: compound--protein binding affinity, protein sequences, ligand descriptors.
- GraphVAE/MolGAN/JT-VAE: chemically valid molecular graph generation.
- DeepDDS: drug synergy and cell-line response.
- HiGNN: molecular property prediction with atom/fragment hierarchy and attention.

Free resources:

- [OpenStax Organic Chemistry](https://openstax.org/details/books/organic-chemistry): strong open textbook; use chapters on bonding, stereochemistry, aromaticity, carbonyls, amines, biomolecules.
- [MIT OpenCourseWare Chemistry](https://ocw.mit.edu/search/?d=Chemistry): organic, biological, and quantum chemistry lecture material.
- [LibreTexts Chemistry](https://chem.libretexts.org/): useful for focused refreshers.
- [Organic Chemistry with a Biological Emphasis](https://digitalcommons.morris.umn.edu/chem_facpubs/1/) by Tim Soderberg: open organic text with biochemical framing.

Because open textbook PDFs are re-rendered over time, this notebook cites chapters/sections instead of brittle page numbers. If your local PDF has fixed pagination, annotate the page numbers directly in the markdown cells as you study.
"""),
    code(r"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
    HAS_RDKIT = True
except Exception as exc:
    HAS_RDKIT = False
    RDKIT_IMPORT_ERROR = repr(exc)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "Learning" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "Learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("RDKit available:", HAS_RDKIT)
"""),
    md(r"""
## Lesson 1 — What a molecule graph forgets and what it remembers

Most notebooks represent a molecule as a graph:

- nodes: atoms;
- edges: bonds;
- node features: element, degree, formal charge, aromaticity, hybridization, chirality;
- edge features: bond order/type, conjugation, ring membership, stereo.

This representation remembers connectivity well, but it can under-represent:

- conformational ensembles;
- solvent and pH/protonation state;
- tautomeric state;
- long-range electrostatics;
- protein environment;
- quantum electronic structure.

That gap explains why learned molecular representations are powerful but never magic. The model is only as chemically awake as the representation lets it be.
"""),
    code(r"""
if HAS_RDKIT:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "C[C@H](N)C(=O)O"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    display(Draw.MolsToGridImage(mols, legends=smiles, molsPerRow=4, subImgSize=(220, 180)))
else:
    print(RDKIT_IMPORT_ERROR)
"""),
    md(r"""
## Lesson 2 — Bonding, hybridization, and geometry

Useful mental model:

- `sp3`: tetrahedral, about 109.5 degrees, usually single bonds.
- `sp2`: trigonal planar, about 120 degrees, often pi systems/carbonyls/aromatics.
- `sp`: linear, about 180 degrees, alkynes/nitriles.

Why ML cares:

- Atom features often include hybridization.
- 3D quantum properties depend on geometry, not just graph topology.
- Conformer generation in RDKit is a classical approximation to a quantum/thermodynamic reality.
"""),
    code(r"""
angles = np.linspace(80, 190, 400)
ideal = {"sp3 tetrahedral": 109.5, "sp2 trigonal": 120.0, "sp linear": 180.0}
for name, mu in ideal.items():
    y = np.exp(-0.5 * ((angles - mu) / 6) ** 2)
    plt.plot(angles, y, label=name)
plt.xlabel("bond angle / degrees")
plt.ylabel("schematic stability")
plt.title("Idealized angle preferences")
plt.legend()
plt.show()
"""),
    md(r"""
## Lesson 3 — Aromaticity and conjugation

Aromatic systems are cyclic, planar, conjugated, and often follow Hückel's $4n+2$ pi-electron rule.

Why ML cares:

- RDKit marks aromatic atoms/bonds.
- Aromatic rings dominate drug-like scaffolds.
- Fragment models such as JT-VAE and HiGNN often preserve ring systems as meaningful substructures.
"""),
    code(r"""
if HAS_RDKIT:
    examples = {
        "benzene": "c1ccccc1",
        "pyridine": "n1ccccc1",
        "cyclohexane": "C1CCCCC1",
        "imidazole": "c1ncc[nH]1",
    }
    rows = []
    for name, smi in examples.items():
        mol = Chem.MolFromSmiles(smi)
        rows.append({
            "name": name,
            "smiles": smi,
            "aromatic_atoms": sum(a.GetIsAromatic() for a in mol.GetAtoms()),
            "aromatic_bonds": sum(b.GetIsAromatic() for b in mol.GetBonds()),
            "rings": rdMolDescriptors.CalcNumRings(mol),
        })
    display(pd.DataFrame(rows))
    display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in examples.values()], legends=list(examples), subImgSize=(220, 180)))
"""),
    md(r"""
## Lesson 4 — Stereochemistry and chirality

Two molecules can share the same graph and differ in 3D arrangement. Protein binding pockets are chiral; stereochemistry can decide activity, toxicity, and metabolism.

Key distinctions:

- enantiomers: non-superimposable mirror images;
- diastereomers: stereoisomers that are not mirror images;
- E/Z alkene stereochemistry;
- conformers: interconverting 3D shapes, not different connectivity.

Why ML cares:

- SMILES may or may not encode chirality (`@`, `@@`, `/`, `\\`).
- Atom/bond features in MoleculeNet often include chirality/stereo fields.
- Generative models must avoid producing impossible or unspecified stereochemistry when the task cares.
"""),
    code(r"""
if HAS_RDKIT:
    chiral = ["C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"]
    mols = [Chem.MolFromSmiles(s) for s in chiral]
    display(Draw.MolsToGridImage(mols, legends=chiral, subImgSize=(260, 200)))
    for s, m in zip(chiral, mols):
        centers = Chem.FindMolChiralCenters(m, includeUnassigned=True)
        print(s, centers)
"""),
    md(r"""
## Lesson 5 — Physicochemical properties used in notebooks and papers

Common learned or provided properties:

- molecular weight: size proxy;
- LogP: lipophilicity, membrane permeability vs solubility tension;
- TPSA: polar surface area, permeability/solvation proxy;
- H-bond donors/acceptors: binding and solubility;
- rotatable bonds: flexibility/entropy;
- formal charge/protonation: electrostatics and pH dependence;
- QED/drug-likeness: heuristic desirability;
- solubility/free energy/lipophilicity labels in MoleculeNet-style datasets.

Lipinski's rule of five is a filter, not a law:

$$
MW < 500,\quad \log P < 5,\quad HBD \le 5,\quad HBA \le 10.
$$
"""),
    code(r"""
if HAS_RDKIT:
    drugs = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
        "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "imatinib fragment-ish": "Cc1ccc(NC(=O)c2ccc(CN3CCNCC3)cc2)cc1",
    }
    rows = []
    for name, smi in drugs.items():
        m = Chem.MolFromSmiles(smi)
        rows.append({
            "name": name,
            "MW": Descriptors.MolWt(m),
            "LogP": Descriptors.MolLogP(m),
            "TPSA": rdMolDescriptors.CalcTPSA(m),
            "HBD": rdMolDescriptors.CalcNumHBD(m),
            "HBA": rdMolDescriptors.CalcNumHBA(m),
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "rings": rdMolDescriptors.CalcNumRings(m),
        })
    df = pd.DataFrame(rows)
    display(df.round(2))
    df.set_index("name")[["MW", "LogP", "TPSA", "HBD", "HBA", "rot_bonds"]].plot(kind="bar", figsize=(10, 4))
    plt.title("Descriptor scales differ; standardize before ML")
    plt.tight_layout()
    plt.show()
"""),
    md(r"""
## Lesson 6 — Quantum chemistry vocabulary for MPNNs

The quantum-chemistry labels in datasets like QM9 are approximations to molecular electronic structure.

Core concepts:

- electronic wavefunction $\Psi$ contains the quantum state;
- Hamiltonian $\hat{H}$ is the energy operator;
- Schrödinger equation: $\hat{H}\Psi = E\Psi$;
- Born--Oppenheimer approximation: nuclei are slow, electrons adjust quickly;
- HOMO/LUMO: highest occupied / lowest unoccupied molecular orbital;
- HOMO--LUMO gap: rough chemical reactivity/electronic excitation proxy;
- dipole moment: charge separation;
- atomization energy: energy to separate molecule into atoms.

Why MPNNs work at all: local chemical environments strongly influence many molecular properties, so message passing can approximate the map from graph/geometry to property.
"""),
    code(r"""
# Toy one-dimensional orbital shapes: not quantum chemistry software, just intuition.
x = np.linspace(-4, 4, 500)
psi_1s = np.exp(-np.abs(x))
psi_2p = x * np.exp(-np.abs(x))
plt.plot(x, psi_1s / np.linalg.norm(psi_1s), label="schematic bonding-like orbital")
plt.plot(x, psi_2p / np.linalg.norm(psi_2p), label="schematic antibonding/node-like orbital")
plt.axhline(0, color="black", lw=0.5)
plt.title("Orbital intuition: nodes usually imply higher energy")
plt.xlabel("position")
plt.ylabel("amplitude")
plt.legend()
plt.show()
"""),
    md(r"""
## Lesson 7 — Protein--ligand binding affinity

DeepAffinity-style models connect a compound representation with a protein representation.

Thermodynamic relation:

$$
\Delta G = RT \ln K_d
$$

or, using molar units at room temperature, a lower $K_d$ means tighter binding and more negative $\Delta G$.

Chemistry and biology behind affinity:

- shape complementarity;
- hydrogen bonds;
- salt bridges/electrostatics;
- hydrophobic effect;
- pi stacking/cation-pi interactions;
- desolvation penalties;
- conformational entropy;
- induced fit/allostery.

Important warning: sequence-only protein encoders miss 3D binding-site geometry unless the model/data indirectly teaches it.
"""),
    code(r"""
R = 8.314e-3  # kJ mol^-1 K^-1
T = 298.15
Kd = np.logspace(-12, -3, 200)  # M
dG = R * T * np.log(Kd)
plt.semilogx(Kd, dG)
plt.gca().invert_xaxis()
plt.xlabel("Kd / M (left = tighter binding)")
plt.ylabel("Delta G / kJ mol$^{-1}$")
plt.title("Binding affinity scale")
plt.show()
"""),
    md(r"""
## Lesson 8 — Drug synergy and dose response

DeepDDS predicts whether two drugs act synergistically in a cellular context.

Basic vocabulary:

- additive: combined effect equals expectation from single drugs;
- synergistic: combined effect exceeds expectation;
- antagonistic: combined effect is weaker than expectation;
- Bliss/Loewe/HSA/ZIP: different mathematical baselines for expected combination effect.

Why chemistry alone is not enough: synergy depends on targets, pathways, cell-line omics, dosing, transporters, metabolism, and feedback loops.
"""),
    code(r"""
dose_a = np.linspace(0, 5, 80)
dose_b = np.linspace(0, 5, 80)
A, B = np.meshgrid(dose_a, dose_b)
Ea = 1 / (1 + np.exp(-(A - 2.5)))
Eb = 1 / (1 + np.exp(-(B - 2.5)))
bliss = Ea + Eb - Ea * Eb
synergy_bonus = 0.18 * np.exp(-((A - 2.7) ** 2 + (B - 2.7) ** 2) / 1.2)
observed = np.clip(bliss + synergy_bonus, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(bliss, origin="lower", extent=[0, 5, 0, 5], aspect="auto")
axes[0].set_title("Bliss expected effect")
im1 = axes[1].imshow(observed - bliss, origin="lower", extent=[0, 5, 0, 5], aspect="auto", cmap="coolwarm")
axes[1].set_title("Synthetic synergy excess")
for ax in axes:
    ax.set_xlabel("drug A dose")
    ax.set_ylabel("drug B dose")
plt.colorbar(im0, ax=axes[0])
plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.show()
"""),
    md(r"""
## Lesson 9 — Fragment chemistry and molecular generation

Generative models need constraints. Organic molecules are not arbitrary graphs:

- typical valences must be satisfied;
- rings have strain/aromaticity rules;
- functional groups bring predictable reactivity;
- charged states and salts require careful normalization;
- synthetic accessibility is not guaranteed by validity.

JT-VAE uses chemically meaningful substructures/cliques. HiGNN uses fragments to add hierarchy. BRICS-like fragmentation reflects medicinal-chemistry intuition: molecules are built from linkable pieces.
"""),
    code(r"""
if HAS_RDKIT:
    from rdkit.Chem import BRICS
    smi = "CC(=O)Nc1ccc(OCCN2CCOCC2)cc1"
    mol = Chem.MolFromSmiles(smi)
    frags = sorted(BRICS.BRICSDecompose(mol))
    print("BRICS fragments:")
    for frag in frags:
        print(" ", frag)
    display(Draw.MolToImage(mol, size=(450, 250)))
"""),
    md(r"""
## Lesson 10 — Chemistry reading checklist for every molecular ML paper

Ask:

1. What chemical state is represented: neutralized, protonated, tautomer-specific, stereospecific?
2. Are labels quantum-calculated, experimental, assay-derived, or database-curated?
3. Does the representation include 3D geometry?
4. Does the split test scaffold generalization?
5. Are invalid generated molecules filtered after the fact?
6. Are reported attention weights chemically plausible or merely diagnostic?
7. Does the model learn chemistry, dataset bias, or both?

This is where your bioinformatics instincts help: data provenance is biology's quiet dragon, and chemistry has the same dragon wearing a lab coat.
"""),
])


CHEMINFORMATICS = notebook([
    md(r"""
# Cheminformatics: RDKit and molecular representations for the project notebooks

Role of this notebook: I am your cheminformatics tutor. We assume you learned the theory in `AdvancedChemistry`. Here we focus on tools and methods: RDKit, SMILES, descriptors, fingerprints, scaffold splits, graph conversion, fragments, conformers, and ML-ready featurization.

Primary resources:

- [RDKit Getting Started in Python](https://www.rdkit.org/docs/GettingStartedInPython.html)
- [RDKit documentation index](https://www.rdkit.org/docs/index.html)
- [TeachOpenCADD](https://projects.volkamerlab.org/teachopencadd/): excellent open cheminformatics/drug-discovery tutorials.
- [Molecular descriptors overview](https://en.wikipedia.org/wiki/Molecular_descriptor): quick taxonomy; for production work, prefer RDKit docs and primary literature.

Connection to project notebooks:

- MPNNs/DeepDDS/HiGNN need molecular graphs.
- DeepAffinity needs compound encodings and protein sequence encodings.
- GraphVAE/MolGAN/JT-VAE need validity, graph tensors, fragments, and generation metrics.
"""),
    code(r"""
from __future__ import annotations

from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, BRICS, Descriptors, Draw, QED, rdMolDescriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except Exception as exc:
    HAS_RDKIT = False
    RDKIT_IMPORT_ERROR = repr(exc)

try:
    import torch
    HAS_TORCH = True
except Exception as exc:
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = repr(exc)

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except Exception as exc:
    HAS_PYG = False
    PYG_IMPORT_ERROR = repr(exc)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "Learning" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "Learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("RDKit:", HAS_RDKIT)
print("Torch:", HAS_TORCH)
print("PyG:", HAS_PYG)
"""),
    md(r"""
## Lesson 1 — SMILES in, molecule object out

RDKit's central object is `Mol`. A SMILES string is a compact text representation; a `Mol` is a parsed chemical graph with atoms, bonds, valence checks, aromaticity perception, stereochemistry, and properties.

Key habit: always check whether parsing returned `None`.
"""),
    code(r"""
if not HAS_RDKIT:
    raise RuntimeError(RDKIT_IMPORT_ERROR)

smiles = ["CCO", "c1ccccc1", "CO(C)C", "C[C@H](N)C(=O)O"]
for s in smiles:
    mol = Chem.MolFromSmiles(s)
    print(f"{s:20s}", "OK" if mol is not None else "FAILED")

valid = [Chem.MolFromSmiles(s) for s in smiles if Chem.MolFromSmiles(s) is not None]
display(Draw.MolsToGridImage(valid, legends=[Chem.MolToSmiles(m) for m in valid], subImgSize=(220, 180)))
"""),
    md(r"""
## Lesson 2 — Canonicalization, sanitization, and why the same molecule has many strings

The same molecule can be written many ways. Canonical SMILES helps deduplicate. Isomeric SMILES preserves stereochemistry when present.
"""),
    code(r"""
equiv = ["C1=CC=CN=C1", "c1cccnc1", "n1ccccc1"]
for s in equiv:
    m = Chem.MolFromSmiles(s)
    print(s, "->", Chem.MolToSmiles(m), "| isomeric:", Chem.MolToSmiles(m, isomericSmiles=True))
"""),
    md(r"""
## Lesson 3 — Inspecting atoms and bonds

These are the raw ingredients for graph neural network features.
"""),
    code(r"""
m = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
atom_rows = []
for a in m.GetAtoms():
    atom_rows.append({
        "idx": a.GetIdx(),
        "symbol": a.GetSymbol(),
        "atomic_num": a.GetAtomicNum(),
        "degree": a.GetDegree(),
        "formal_charge": a.GetFormalCharge(),
        "hybridization": str(a.GetHybridization()),
        "aromatic": a.GetIsAromatic(),
        "in_ring": a.IsInRing(),
    })
bond_rows = []
for b in m.GetBonds():
    bond_rows.append({
        "begin": b.GetBeginAtomIdx(),
        "end": b.GetEndAtomIdx(),
        "type": str(b.GetBondType()),
        "conjugated": b.GetIsConjugated(),
        "aromatic": b.GetIsAromatic(),
        "in_ring": b.IsInRing(),
    })
display(pd.DataFrame(atom_rows))
display(pd.DataFrame(bond_rows).head())
display(Draw.MolToImage(m, size=(450, 250)))
"""),
    md(r"""
## Lesson 4 — Descriptors: human-designed molecular numbers

Descriptors are useful baselines and sanity checks. Deep models may learn better features, but descriptors reveal whether the data distribution makes chemical sense.
"""),
    code(r"""
library = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
    "imatinib_core": "Cc1ccc(NC(=O)c2ccc(CN3CCNCC3)cc2)cc1",
}
rows = []
for name, smi in library.items():
    mol = Chem.MolFromSmiles(smi)
    rows.append({
        "name": name,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "QED": QED.qed(mol),
    })
df = pd.DataFrame(rows)
display(df.round(3))
df.to_csv(DATA_DIR / "toy_molecule_descriptors.csv", index=False)
print("Saved:", DATA_DIR / "toy_molecule_descriptors.csv")
"""),
    md(r"""
## Lesson 5 — Fingerprints and similarity

Morgan fingerprints, RDKit's implementation of circular fingerprints, are the workhorse representation behind many QSAR baselines.

Tanimoto similarity for bit vectors:

$$
T(A,B) = \frac{|A \cap B|}{|A \cup B|}.
$$

This is also a useful way to diagnose train/test leakage: if your test molecules are near-duplicates of train molecules, random-split performance can look heroic for boring reasons.
"""),
    code(r"""
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mols = [Chem.MolFromSmiles(s) for s in library.values()]
fps = [fpgen.GetFingerprint(m) for m in mols]
names = list(library)
sim = np.zeros((len(fps), len(fps)))
for i, fpi in enumerate(fps):
    for j, fpj in enumerate(fps):
        sim[i, j] = DataStructs.TanimotoSimilarity(fpi, fpj)
plt.imshow(sim, vmin=0, vmax=1, cmap="viridis")
plt.xticks(range(len(names)), names, rotation=45, ha="right")
plt.yticks(range(len(names)), names)
plt.colorbar(label="Tanimoto")
plt.title("Morgan fingerprint similarity")
plt.tight_layout()
plt.show()
"""),
    md(r"""
## Lesson 6 — Scaffold splitting

Random splits often overestimate molecular ML performance. Bemis--Murcko scaffolds group molecules by core ring/linker topology. Scaffold splits ask a harder question: can the model generalize to new chemical cores?
"""),
    code(r"""
scaffold_rows = []
for name, smi in library.items():
    mol = Chem.MolFromSmiles(smi)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    scaffold_rows.append({"name": name, "smiles": smi, "scaffold": scaffold})
display(pd.DataFrame(scaffold_rows))
scaffold_mols = [Chem.MolFromSmiles(r["scaffold"]) if r["scaffold"] else Chem.MolFromSmiles(r["smiles"]) for r in scaffold_rows]
display(Draw.MolsToGridImage(scaffold_mols, legends=[r["name"] for r in scaffold_rows], subImgSize=(220, 180)))
"""),
    md(r"""
## Lesson 7 — Conformers and 3D coordinates

RDKit can generate approximate 3D conformers with distance geometry and optimize them with molecular mechanics. This is not quantum chemistry, but it gives useful geometry for descriptors and visualization.
"""),
    code(r"""
mol = Chem.AddHs(Chem.MolFromSmiles(library["aspirin"]))
params = AllChem.ETKDGv3()
params.randomSeed = 42
ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
energies = []
for cid in ids:
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
    ff.Minimize(maxIts=200)
    energies.append(ff.CalcEnergy())
plt.bar(range(len(energies)), energies)
plt.xlabel("conformer id")
plt.ylabel("UFF energy / arbitrary units")
plt.title("Conformer energy spread")
plt.show()
print("Lowest energy conformer:", int(np.argmin(energies)), "energy:", min(energies))
"""),
    md(r"""
## Lesson 8 — Molecule to PyTorch Geometric graph

This is the bridge to MPNNs, DeepDDS, HiGNN, GraphVAE-style encoders, and many graph predictors.

Below is a deliberately simple featurizer. Production notebooks usually use richer categorical encodings.
"""),
    code(r"""
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]

bond_type_to_float = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}

def mol_to_pyg(mol):
    if not (HAS_TORCH and HAS_PYG):
        raise RuntimeError("Requires torch and torch_geometric")
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_pairs = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = [bond_type_to_float.get(b.GetBondType(), 0.0), int(b.GetIsConjugated()), int(b.IsInRing())]
        edge_pairs.extend([(i, j), (j, i)])
        edge_attr.extend([feat, feat])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=Chem.MolToSmiles(mol))

if HAS_TORCH and HAS_PYG:
    data = mol_to_pyg(Chem.MolFromSmiles(library["aspirin"]))
    print(data)
    print("x:\n", data.x[:5])
    print("edge_index shape:", data.edge_index.shape)
else:
    print("Torch/PyG missing:", globals().get("TORCH_IMPORT_ERROR"), globals().get("PYG_IMPORT_ERROR"))
"""),
    md(r"""
## Lesson 9 — Fragments for HiGNN and JT-VAE intuition

Fragmentation turns a molecule into chemically meaningful pieces. In this project:

- HiGNN uses an atom graph plus fragment-level hierarchy.
- JT-VAE uses a junction tree of molecular substructures to make generation more chemically valid.
"""),
    code(r"""
target = Chem.MolFromSmiles("CC(=O)Nc1ccc(OCCN2CCOCC2)cc1")
frags = sorted(BRICS.BRICSDecompose(target))
frag_mols = [Chem.MolFromSmiles(f) for f in frags]
print("Number of BRICS fragments:", len(frags))
for f in frags:
    print(f)
display(Draw.MolsToGridImage([target] + frag_mols, legends=["parent"] + frags, molsPerRow=3, subImgSize=(260, 180)))
"""),
    md(r"""
## Lesson 10 — Simple validity, uniqueness, novelty metrics for generated molecules

Generative-model papers often report:

- validity: fraction of generated SMILES that parse and sanitize;
- uniqueness: fraction of valid molecules that are non-duplicates;
- novelty: fraction of valid molecules not in the training set;
- property distribution match: QED, LogP, MW, SA-like score, etc.

Validity is necessary but weak. `CCCCCCCCCCCCCCCC` can be valid and boring. A model can be valid, unique, and useless if it ignores the desired chemical space.
"""),
    code(r"""
generated = [
    "CCO", "OCC", "c1ccccc1", "C1CC1", "CO(C)C", "not_a_smiles",
    "CC(=O)O", "CC(=O)O", "C[N+](C)(C)C",
]
train = {"CCO", "c1ccccc1"}

valid_mols = []
valid_smiles = []
for smi in generated:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        can = Chem.MolToSmiles(mol)
        valid_mols.append(mol)
        valid_smiles.append(can)

metrics = {
    "n_generated": len(generated),
    "n_valid": len(valid_smiles),
    "validity": len(valid_smiles) / len(generated),
    "uniqueness": len(set(valid_smiles)) / max(1, len(valid_smiles)),
    "novelty": len([s for s in set(valid_smiles) if s not in train]) / max(1, len(set(valid_smiles))),
}
print(metrics)
display(Draw.MolsToGridImage(valid_mols, legends=valid_smiles, molsPerRow=4, subImgSize=(200, 160)))
"""),
    md(r"""
## Lesson 11 — Protein sequences for DeepAffinity

RDKit handles compounds, not proteins as sequences. For DeepAffinity-like models, a common starting point is amino-acid tokenization:

- map 20 canonical residues plus unknown/padding to integers;
- embed tokens;
- use CNN/RNN/Transformer/attention pooling;
- combine protein representation with compound representation.

The cheminformatics trap: a compound--protein affinity model has two different representation worlds. Bad preprocessing on either side can dominate performance.
"""),
    code(r"""
aa = "ACDEFGHIKLMNPQRSTVWY"
stoi = {ch: i + 1 for i, ch in enumerate(aa)}
stoi["X"] = len(stoi) + 1
PAD = 0

def tokenize_protein(seq, max_len=32):
    arr = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        arr[i] = stoi.get(ch, stoi["X"])
    return arr

seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVA"
tokens = tokenize_protein(seq)
print(tokens)
print("length before truncation:", len(seq), "encoded length:", len(tokens))
"""),
    md(r"""
## Lesson 12 — A practical preprocessing checklist

Before training molecular ML models:

1. Parse SMILES and log failures.
2. Canonicalize duplicates.
3. Decide what to do with salts/mixtures.
4. Decide protonation/tautomer policy.
5. Preserve stereochemistry if labels depend on it.
6. Compute descriptors for sanity plots.
7. Use scaffold/time/assay-aware splits where appropriate.
8. Save processed inputs to `data/`.
9. Save model weights/checkpoints to `models/`.
10. Record package versions with `Environment_Hardware_Check.ipynb`.

This is the unglamorous layer where many model papers quietly win or lose.
"""),
])


def expand_notebooks() -> None:
    """Add the more detailed second-pass curriculum cells.

    The initial notebooks intentionally established a runnable spine. These
    cells turn that spine into a fuller guided course while keeping each
    notebook executable on a normal Python/RDKit/PyTorch environment.
    """

    deep_extra = [
        md(r"""
## Lesson 11 — Dataset design, leakage, and why molecular ML is unusually sneaky

In image classification, a random train/test split is often defensible. In molecular ML, it can be dangerously optimistic because close analogs may appear in both train and test. A model can then learn a local interpolation rule rather than chemistry that transfers to new scaffolds.

Important split types:

| split | what it tests | typical use |
|---|---|---|
| random | interpolation among similar molecules | quick debugging |
| scaffold | generalization to new cores | molecular property prediction |
| time split | future compounds from past compounds | medicinal chemistry programs |
| protein-family split | new protein classes | affinity/generalization |
| cell-line split | new biological contexts | drug response/synergy |

Leakage examples in this repo's domain:

- same compound appears with slightly different salts/protonation;
- stereochemistry stripped from one copy but not another;
- assay replicates split across train/test;
- protein sequence near-duplicates split across train/test;
- generated molecule novelty measured against canonical SMILES instead of standardized molecules.

Research habit: before tuning the model, plot distributions and nearest-neighbor similarities between splits.
"""),
        code(r"""
# A tiny visual intuition for random vs. grouped splitting.
rng = np.random.default_rng(42)
centers = np.array([[-2, -1], [0, 1], [2, -1]])
Xtoy, groups = [], []
for gid, c in enumerate(centers):
    pts = c + 0.35 * rng.normal(size=(40, 2))
    Xtoy.append(pts)
    groups += [gid] * len(pts)
Xtoy = np.vstack(Xtoy)
groups = np.array(groups)

random_test = rng.choice(len(Xtoy), size=24, replace=False)
group_test = np.where(groups == 2)[0]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, test_idx, title in [(axes[0], random_test, "random split"), (axes[1], group_test, "scaffold-like group split")]:
    train_mask = np.ones(len(Xtoy), dtype=bool)
    train_mask[test_idx] = False
    ax.scatter(Xtoy[train_mask, 0], Xtoy[train_mask, 1], s=20, label="train", alpha=0.6)
    ax.scatter(Xtoy[test_idx, 0], Xtoy[test_idx, 1], s=35, label="test", alpha=0.9)
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
plt.suptitle("Generalization is defined by the split, not the optimizer")
plt.tight_layout()
plt.show()
"""),
        md(r"""
## Lesson 12 — Metrics you will see in the project notebooks

Regression:

$$
\mathrm{MAE}=\frac{1}{N}\sum_i |\hat{y}_i-y_i|,\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(\hat{y}_i-y_i)^2}
$$

RMSE punishes outliers more strongly. MAE is often easier to interpret chemically if the units are meaningful.

Binary classification:

- accuracy: easy but bad for imbalanced data;
- precision/recall: useful for active-hit discovery;
- ROC-AUC: probability a random positive ranks above a random negative;
- PR-AUC: better under severe class imbalance.

Generative models:

- validity, uniqueness, novelty;
- reconstruction accuracy for VAEs;
- distribution matching for descriptors;
- property optimization;
- diversity and scaffold diversity.

For molecular science, always pair metrics with examples. A beautiful ROC-AUC plus chemically absurd false positives is not a win; it is a warning siren with a violin section.
"""),
        code(r"""
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "RMSE": rmse}

def roc_auc_rank(y_true, score):
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score)
    pos = score[y_true]
    neg = score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return wins / (len(pos) * len(neg))

y_true = np.array([0.0, 1.0, 2.0, 4.0])
print(regression_metrics(y_true, [0.1, 1.2, 1.8, 7.0]))
print("ROC-AUC:", roc_auc_rank([0, 1, 0, 1, 1], [0.1, 0.9, 0.4, 0.6, 0.7]))
"""),
        md(r"""
## Lesson 13 — Batching variable-size objects

Molecules, proteins, and drug pairs rarely have the same size:

- graphs have different numbers of atoms/bonds;
- proteins have different sequence lengths;
- generated graphs may have padded maximum sizes;
- junction trees have different numbers of cliques.

Two standard strategies:

1. **Padding + mask** for sequences or fixed-size graph tensors.
2. **Concatenation + index vectors** for PyG graph batches.

Masking matters. If you mean-pool padded tokens without a mask, the model learns that sequence length and padding are chemistry. That is a very small goblin, but it can wreck a result.
"""),
        code(r"""
# Sequence padding with a mask.
lengths = torch.tensor([5, 9, 3])
max_len = int(lengths.max())
emb_dim = 4
seq = torch.zeros(len(lengths), max_len, emb_dim)
for i, L in enumerate(lengths):
    seq[i, :L] = torch.randn(L, emb_dim)
mask = torch.arange(max_len)[None, :] < lengths[:, None]
wrong_pool = seq.mean(dim=1)
right_pool = (seq * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
print("wrong pooled shape:", wrong_pool.shape)
print("right pooled shape:", right_pool.shape)
print("difference for padded examples:", (wrong_pool - right_pool).norm(dim=1))
"""),
        md(r"""
## Lesson 14 — A reusable PyTorch training loop pattern

Almost every supervised notebook can be reduced to:

1. create dataset and split;
2. make dataloaders;
3. initialize model;
4. train one epoch;
5. evaluate;
6. save best checkpoint;
7. reload checkpoint for test/prediction.

Cluster-specific additions:

- mixed precision with `torch.amp.autocast`;
- `num_workers` and `pin_memory` for GPU data loading;
- checkpoint every epoch or every validation improvement;
- log the git hash/environment report if possible.

The point of a training loop is not just to optimize. It is to make experiments restartable.
"""),
        code(r"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(path, model, optimizer=None, **metadata):
    payload = {"model_state": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)

def load_checkpoint(path, model, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload.get("metadata", {})

demo = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
print("parameters:", count_parameters(demo))
save_checkpoint(MODEL_DIR / "training_loop_demo.pt", demo, note="minimal checkpoint pattern")
print(load_checkpoint(MODEL_DIR / "training_loop_demo.pt", demo))
"""),
        md(r"""
## Lesson 15 — Mixed precision and GPU performance

On modern NVIDIA GPUs, float16/bfloat16 matrix operations can be much faster than float32. PyTorch's automatic mixed precision keeps many operations in lower precision while preserving stability for sensitive operations.

Typical pattern:

```python
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Use mixed precision for large GNNs/sequence models on the cluster. Be cautious with:

- tiny losses and unstable VAEs/GANs;
- custom operations;
- metrics computed in low precision;
- CPU execution, where mixed precision usually does not help.
"""),
        code(r"""
if torch.cuda.is_available():
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter() if "time" in globals() else None
    with torch.amp.autocast("cuda"):
        c = a @ b
    torch.cuda.synchronize()
    print("autocast matmul dtype:", c.dtype)
else:
    print("CUDA not available here; run this cell on the cluster to see mixed-precision behavior.")
"""),
        md(r"""
## Lesson 16 — Oversmoothing and over-squashing in GNNs

Two common GNN failure modes:

**Oversmoothing:** after many message-passing layers, node embeddings become too similar.

**Over-squashing:** long-range information from many nodes is compressed through narrow graph bottlenecks.

Symptoms:

- deeper GNN performs worse;
- node embeddings have low variance;
- model cannot capture long-range functional-group interactions.

Mitigations:

- residual/skip connections;
- normalization;
- virtual nodes/global tokens;
- hierarchical pooling/fragments;
- attention or edge-aware message functions;
- avoid excessive depth unless there is a reason.

This is one reason HiGNN's fragment hierarchy is appealing: it gives information a coarser route through the molecule.
"""),
        code(r"""
if HAS_PYG:
    # Oversmoothing toy: repeatedly apply normalized adjacency-like averaging on a path graph.
    n = 25
    h = torch.eye(n)
    A = torch.zeros(n, n)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    A += torch.eye(n)
    D_inv = torch.diag(1 / A.sum(dim=1))
    P = D_inv @ A
    variances = []
    for _ in range(40):
        h = P @ h
        variances.append(float(h.var(dim=0).mean()))
    plt.plot(variances)
    plt.yscale("log")
    plt.xlabel("message passing steps")
    plt.ylabel("mean feature variance")
    plt.title("Toy oversmoothing by repeated neighbor averaging")
    plt.show()
else:
    print("PyG not required for the math, but earlier imports failed:", globals().get("PYG_IMPORT_ERROR"))
"""),
        md(r"""
## Lesson 17 — Multi-task learning and missing labels

Molecular datasets often contain multiple targets:

- toxicity endpoints;
- quantum properties;
- assay panels;
- cell-line/drug-pair responses.

The loss with missing labels usually uses a mask:

$$
L = \frac{\sum_{i,t} m_{i,t}\ell(\hat{y}_{i,t}, y_{i,t})}{\sum_{i,t}m_{i,t}}.
$$

This appears simple but changes everything:

- batches have different effective target counts;
- rare tasks may be undertrained;
- task scaling matters for regression;
- one task can dominate shared representations.
"""),
        code(r"""
pred = torch.tensor([[0.2, -1.0, 0.4], [1.2, 0.1, -0.5]], requires_grad=True)
target = torch.tensor([[1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32)
loss_per_entry = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
masked_loss = (loss_per_entry * mask).sum() / mask.sum()
masked_loss.backward()
print("masked BCE:", float(masked_loss))
print("gradient exists where label is present; masked positions contribute zero to loss")
"""),
        md(r"""
## Lesson 18 — Reading paper architectures as computation graphs

When a paper diagram looks intimidating, rewrite it as:

1. input tensors;
2. encoder(s);
3. interaction/fusion;
4. pooling/readout;
5. objective.

Examples:

- **MPNN:** atom/bond tensors → message passing → graph readout → property loss.
- **DeepAffinity:** compound encoder + protein encoder → interaction/attention → affinity loss.
- **DeepDDS:** drug A graph + drug B graph + cell-line features → fusion/attention → synergy class.
- **HiGNN:** atom graph + fragment graph → feature-wise attention fusion → property prediction.
- **GraphVAE:** graph encoder → latent Gaussian → padded graph decoder → ELBO/reconstruction.
- **JT-VAE:** molecular graph → junction tree + graph latent variables → constrained decode.
- **MolGAN:** noise → graph generator, graph → discriminator/reward → adversarial/RL losses.

If you can name the tensors at every arrow, you understand the method well enough to implement a faithful reproduction.
"""),
    ]

    chemistry_extra = [
        md(r"""
## Lesson 11 — Acid/base chemistry, protonation, and pH

Many molecular ML failures come from pretending a molecule has one timeless structure. In solution, protonation depends on pH and pKa.

Henderson--Hasselbalch:

For an acid $HA \rightleftharpoons H^+ + A^-$,

$$
\mathrm{pH}=\mathrm{p}K_a+\log_{10}\frac{[A^-]}{[HA]}.
$$

Fraction deprotonated:

$$
\alpha_{A^-}=\frac{1}{1+10^{pK_a-pH}}.
$$

Why ML cares:

- formal charge changes graph features;
- binding pockets select protonation states;
- LogP/logD differ because logD is pH-dependent;
- generated molecules may be valid as graphs but implausible at physiological pH.
"""),
        code(r"""
pH = np.linspace(0, 14, 400)
pKas = [4.5, 7.4, 10.0]
for pKa in pKas:
    frac_deprot = 1 / (1 + 10 ** (pKa - pH))
    plt.plot(pH, frac_deprot, label=f"acid pKa={pKa}")
plt.axvline(7.4, color="black", ls="--", lw=1, label="physiological pH")
plt.xlabel("pH")
plt.ylabel("fraction deprotonated")
plt.title("Acid protonation state depends strongly on pH near pKa")
plt.legend()
plt.show()
"""),
        md(r"""
## Lesson 12 — Tautomers and resonance: same formula, different graph emphasis

Resonance structures are not distinct molecules; they are bookkeeping for delocalized electrons. Tautomers, however, are interconverting constitutional isomers with different atom connectivity/proton placement.

ML consequences:

- canonical SMILES does not automatically solve tautomer equivalence;
- different databases may store different tautomers;
- generated molecules may exploit tautomer edge cases;
- binding affinity may favor one tautomer in a pocket.

RDKit and other toolkits have tautomer enumerators, but tautomer standardization is a modeling decision, not a universal truth.
"""),
        code(r"""
if HAS_RDKIT:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    taut = rdMolStandardize.TautomerEnumerator()
    mol = Chem.MolFromSmiles("CC(=O)NC")  # simple amide; limited tautomerism
    keto_enol = Chem.MolFromSmiles("CC(=O)C")
    tautomers = list(taut.Enumerate(keto_enol))
    print("Enumerated tautomers:", len(tautomers))
    display(Draw.MolsToGridImage(tautomers[:8], legends=[Chem.MolToSmiles(m) for m in tautomers[:8]], subImgSize=(220, 160)))
"""),
        md(r"""
## Lesson 13 — Solvation, hydrophobic effect, and permeability

Binding is not just ligand + protein. It is ligand + protein + water + ions + entropy.

Key ideas:

- Polar groups interact favorably with water but may pay desolvation penalties entering hydrophobic pockets.
- Hydrophobic groups can gain binding affinity by displacing ordered water.
- Membrane permeability often prefers lower polarity, but solubility prefers higher polarity.
- TPSA and LogP are crude but useful proxies.

This is why drug design is full of tradeoffs. A model optimizing one property can easily damage another.
"""),
        code(r"""
mw = np.linspace(150, 650, 120)
tpsa = np.linspace(10, 180, 120)
MW, TPSA = np.meshgrid(mw, tpsa)
# Toy permeability score, not a real model.
score = np.exp(-((TPSA - 55) / 55) ** 2) * np.exp(-((MW - 350) / 180) ** 2)
plt.contourf(MW, TPSA, score, levels=20, cmap="viridis")
plt.colorbar(label="schematic permeability favorability")
plt.xlabel("molecular weight")
plt.ylabel("TPSA")
plt.title("Toy tradeoff: permeability often dislikes high polarity and high size")
plt.show()
"""),
        md(r"""
## Lesson 14 — Reaction/functional group intuition for generated molecules

Generative models can produce molecules that are formally valid but chemically undesirable.

Watch for:

- reactive acyl halides, aldehydes, Michael acceptors;
- strained rings;
- unstable peroxides;
- pan-assay interference compounds (PAINS);
- redox-active motifs;
- promiscuous aggregators;
- toxicophores.

The papers here mostly focus on representation/generation, not full medicinal chemistry triage. When you use generated molecules seriously, add filters and expert review.
"""),
        code(r"""
if HAS_RDKIT:
    smarts = {
        "aldehyde": "[CX3H1](=O)[#6]",
        "acyl_chloride": "C(=O)Cl",
        "michael_acceptor": "C=CC=O",
        "nitro": "[NX3](=O)=O",
    }
    examples = {
        "benzaldehyde": "O=Cc1ccccc1",
        "acetyl chloride": "CC(=O)Cl",
        "methyl vinyl ketone": "C=CC(=O)C",
        "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    }
    rows = []
    for name, smi in examples.items():
        mol = Chem.MolFromSmiles(smi)
        row = {"name": name, "smiles": smi}
        for label, patt in smarts.items():
            row[label] = mol.HasSubstructMatch(Chem.MolFromSmarts(patt))
        rows.append(row)
    display(pd.DataFrame(rows))
"""),
        md(r"""
## Lesson 15 — From molecular orbitals to descriptors

Quantum descriptors in QM9-like work include:

- $\mu$: dipole moment;
- $\alpha$: isotropic polarizability;
- $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$;
- gap $\Delta\epsilon = \epsilon_{LUMO}-\epsilon_{HOMO}$;
- electronic spatial extent;
- zero-point vibrational energy;
- internal energies/free energies/enthalpies;
- heat capacity.

Message-passing models approximate these from atoms, bonds, and sometimes 3D distances. Without 3D geometry, stereochemical and conformational effects become hard or impossible to distinguish.
"""),
        code(r"""
# Schematic energy-level diagram.
levels = [-10.2, -8.5, -6.0, -1.8, 0.7]
labels = ["core", "sigma", "HOMO", "LUMO", "higher"]
for e, lab in zip(levels, labels):
    plt.hlines(e, 0, 1, lw=3)
    plt.text(1.05, e, lab, va="center")
plt.annotate("gap", xy=(0.5, -1.8), xytext=(0.5, -6.0), arrowprops={"arrowstyle": "<->"})
plt.xlim(-0.1, 1.8)
plt.ylabel("orbital energy / arbitrary")
plt.xticks([])
plt.title("HOMO/LUMO gap intuition")
plt.show()
"""),
        md(r"""
## Lesson 16 — Assays, noise, and biological labels

A chemical structure is clean; a biological label is often messy.

Sources of label noise:

- batch effects;
- assay format differences;
- protein construct differences;
- cell-line drift;
- compound purity/degradation;
- solubility/aggregation;
- censored values such as `>10 µM`;
- different activity units: IC50, Ki, Kd, EC50.

Affinity conversion:

$$
pK_d = -\log_{10}(K_d \text{ in molar})
$$

Small numerical differences can matter: one log unit is a tenfold affinity change.
"""),
        code(r"""
Kd_uM = np.array([100, 10, 1, 0.1, 0.01])
pKd = -np.log10(Kd_uM * 1e-6)
pd.DataFrame({"Kd_uM": Kd_uM, "pKd": pKd})
"""),
        md(r"""
## Lesson 17 — Chemical validity vs. synthesizability vs. usefulness

Validity: RDKit can parse and sanitize the molecule.

Synthesizability: a chemist can reasonably make it.

Usefulness: it has the desired activity, selectivity, ADMET profile, novelty, and IP position.

These are nested only loosely. A valid molecule may be impossible to synthesize; a synthesizable molecule may be toxic; a potent molecule may be insoluble. Generative models often optimize the first layer because it is easy to measure.
"""),
        md(r"""
## Lesson 18 — Chemistry map of the repo

- **MPNNs:** bonding, atom environments, quantum descriptors, graph locality.
- **DeepAffinity:** ligand properties, protein binding thermodynamics, sequence/protein representation.
- **GraphVAE:** graph validity, atom/bond categorical distributions, padded graph tensors.
- **JT-VAE:** rings, fragments, cliques, synthetic/structural priors.
- **DeepDDS:** pharmacology, synergy, pathway context, drug-pair interactions.
- **MolGAN:** validity/uniqueness/novelty, reward design, property optimization.
- **HiGNN:** BRICS/fragments, hierarchical structure, property-relevant substructures.
"""),
    ]

    cheminf_extra = [
        md(r"""
## Lesson 13 — Standardization: salts, charges, fragments, and neutralization

Before modeling, decide what a "molecule" means.

Common preprocessing:

- keep largest organic fragment;
- remove salts/solvents;
- normalize charges;
- canonicalize tautomers when appropriate;
- preserve or discard stereochemistry deliberately;
- reject mixtures if the model expects one molecular graph.

This is not clerical. Standardization changes the model input and sometimes the label meaning.
"""),
        code(r"""
from rdkit.Chem.MolStandardize import rdMolStandardize

examples = ["CC(=O)[O-].[Na+]", "Cl.CN1CCN(CC1)c1ncccn1", "CCO.O"]
lfc = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()
rows = []
for smi in examples:
    mol = Chem.MolFromSmiles(smi)
    largest = lfc.choose(mol)
    uncharged = uncharger.uncharge(largest)
    rows.append({
        "input": smi,
        "largest_fragment": Chem.MolToSmiles(largest),
        "uncharged_largest": Chem.MolToSmiles(uncharged),
    })
display(pd.DataFrame(rows))
"""),
        md(r"""
## Lesson 14 — Substructure search with SMARTS

SMARTS is a query language for molecular patterns. Use it for:

- functional-group filters;
- dataset audits;
- scaffold/series analysis;
- excluding obvious reactive groups;
- interpreting attention hits.

SMARTS can be subtle; test patterns on positive and negative examples before trusting them.
"""),
        code(r"""
patterns = {
    "phenyl": "c1ccccc1",
    "carboxylic_acid": "C(=O)[OH]",
    "tertiary_amine": "[NX3;H0;!$(NC=O)]",
}
rows = []
for name, smi in library.items():
    mol = Chem.MolFromSmiles(smi)
    row = {"name": name}
    for label, smarts in patterns.items():
        row[label] = mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))
    rows.append(row)
display(pd.DataFrame(rows))
"""),
        md(r"""
## Lesson 15 — Highlighting substructures and atom importance

Many interpretability plots eventually become "which atoms/bonds mattered?" RDKit can highlight atoms from a SMARTS match, attention score, gradient attribution, or fragment membership.
"""),
        code(r"""
mol = Chem.MolFromSmiles(library["aspirin"])
patt = Chem.MolFromSmarts("C(=O)O")
match = mol.GetSubstructMatch(patt)
print("carboxyl match atom indices:", match)
display(Draw.MolToImage(mol, highlightAtoms=list(match), size=(420, 260)))
"""),
        md(r"""
## Lesson 16 — Similarity search and nearest-neighbor baselines

Before celebrating a deep model, compare it to a fingerprint nearest-neighbor baseline. If test molecules are close to train molecules, a simple baseline may perform surprisingly well.

For a regression label, a simple baseline is:

1. compute train/test fingerprints;
2. find nearest train molecule by Tanimoto;
3. predict that train molecule's label.

This is not fancy. That is exactly why it is useful.
"""),
        code(r"""
toy_labels = np.array([0.1, 0.4, 1.2, 1.0, 2.5])
query = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin, aromatic ester acid
qfp = fpgen.GetFingerprint(query)
sims = np.array([DataStructs.TanimotoSimilarity(qfp, fp) for fp in fps])
best = int(sims.argmax())
print("nearest:", names[best], "similarity:", sims[best], "predicted label:", toy_labels[best])
plt.bar(names, sims)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Tanimoto to query")
plt.title("Nearest-neighbor baseline intuition")
plt.tight_layout()
plt.show()
"""),
        md(r"""
## Lesson 17 — Molecular graph tensors for generative models

GraphVAE and MolGAN-style models often use padded dense tensors:

- node tensor: `[max_nodes, atom_types]`;
- adjacency/edge tensor: `[bond_types, max_nodes, max_nodes]`;
- mask: which node slots are real.

Pros:

- simple neural decoder output shape;
- easy batching.

Cons:

- fixed maximum graph size;
- many invalid outputs;
- permutation problem: graph node ordering is arbitrary.
"""),
        code(r"""
atom_vocab = ["C", "N", "O", "F", "PAD"]
bond_vocab = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC]

def dense_graph_tensors(mol, max_nodes=12):
    atom_to_idx = {a: i for i, a in enumerate(atom_vocab)}
    x = np.zeros((max_nodes, len(atom_vocab)), dtype=np.float32)
    edge = np.zeros((len(bond_vocab), max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros(max_nodes, dtype=bool)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if i >= max_nodes:
            continue
        sym = atom.GetSymbol()
        x[i, atom_to_idx.get(sym, atom_to_idx["PAD"])] = 1
        mask[i] = True
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < max_nodes and j < max_nodes and bond.GetBondType() in bond_vocab:
            k = bond_vocab.index(bond.GetBondType())
            edge[k, i, j] = edge[k, j, i] = 1
    x[~mask, atom_to_idx["PAD"]] = 1
    return x, edge, mask

x_dense, e_dense, mask = dense_graph_tensors(Chem.MolFromSmiles(library["caffeine"]), max_nodes=16)
print("node tensor:", x_dense.shape, "edge tensor:", e_dense.shape, "mask:", mask.shape, "real nodes:", mask.sum())
"""),
        md(r"""
## Lesson 18 — Matching generated graphs back to molecules

Decoding graph tensors is harder than encoding:

1. choose atom types;
2. choose bond types;
3. remove padding;
4. build RDKit `RWMol`;
5. sanitize;
6. reject or repair invalid valence/aromaticity.

This is why JT-VAE uses stronger chemical structure. It narrows the decoder's search space to more plausible molecular assemblies.
"""),
        code(r"""
def simple_decode_atoms_bonds(atom_symbols, bonds):
    rw = Chem.RWMol()
    for sym in atom_symbols:
        rw.AddAtom(Chem.Atom(sym))
    for i, j, btype in bonds:
        rw.AddBond(i, j, btype)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        return mol, Chem.MolToSmiles(mol), None
    except Exception as exc:
        return mol, None, repr(exc)

mol_ok, smi_ok, err_ok = simple_decode_atoms_bonds(["C", "C", "O"], [(0, 1, Chem.BondType.SINGLE), (1, 2, Chem.BondType.SINGLE)])
mol_bad, smi_bad, err_bad = simple_decode_atoms_bonds(["C"], [(0, 0, Chem.BondType.SINGLE)])
print("valid decode:", smi_ok, err_ok)
print("invalid decode:", smi_bad, err_bad)
"""),
        md(r"""
## Lesson 19 — Descriptor scaling and feature hygiene

Classic descriptors live on wildly different scales: molecular weight, LogP, TPSA, counts, binary flags. Neural networks usually prefer standardized continuous inputs.

Rules:

- fit scalers on train only;
- transform validation/test with train statistics;
- save scaler parameters;
- never compute global normalization before splitting.
"""),
        code(r"""
cont = df[["MW", "LogP", "TPSA", "HBD", "HBA", "QED"]].copy()
mu = cont.mean()
sigma = cont.std(ddof=0).replace(0, 1)
z = (cont - mu) / sigma
display(z.round(2))
print("means after scaling:")
display(z.mean().round(6))
"""),
        md(r"""
## Lesson 20 — RDKit failure handling and audit logs

Production cheminformatics code should never silently drop molecules. Make an audit table:

- original ID;
- original SMILES;
- parse status;
- standardized SMILES;
- reason for exclusion;
- descriptor sanity checks.

This makes experiments reproducible and defensible.
"""),
        code(r"""
raw = ["CCO", "not_smiles", "C1CC1", "C[N+](C)(C)C", ""]
audit = []
for i, smi in enumerate(raw):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        audit.append({"row": i, "input": smi, "status": "failed_parse", "canonical": None, "MW": None})
    else:
        audit.append({"row": i, "input": smi, "status": "ok", "canonical": Chem.MolToSmiles(mol), "MW": Descriptors.MolWt(mol)})
audit_df = pd.DataFrame(audit)
display(audit_df)
audit_df.to_csv(DATA_DIR / "rdkit_audit_demo.csv", index=False)
print("Saved:", DATA_DIR / "rdkit_audit_demo.csv")
"""),
        md(r"""
## Lesson 21 — What to learn next in cheminformatics

After this notebook, good next steps are:

- TeachOpenCADD talktorials on compound datasets, similarity, docking, and machine learning.
- RDKit UGM notebooks and blog posts by Greg Landrum.
- DeepChem examples for molecular datasets and featurizers.
- Basic docking and structure-based design if you want to connect sequence affinity models to 3D pockets.

The biggest conceptual step is accepting that "SMILES → tensor" is not a neutral act. It is a scientific modeling choice wearing a programmer's hoodie.
"""),
    ]

    DEEP_LEARNING["cells"].extend(deep_extra)
    ADVANCED_CHEMISTRY["cells"].extend(chemistry_extra)
    CHEMINFORMATICS["cells"].extend(cheminf_extra)


def write(name: str, nb: dict) -> None:
    path = ROOT / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    expand_notebooks()
    write("DeepLearning.ipynb", DEEP_LEARNING)
    write("AdvancedChemistry.ipynb", ADVANCED_CHEMISTRY)
    write("Cheminformatics.ipynb", CHEMINFORMATICS)
