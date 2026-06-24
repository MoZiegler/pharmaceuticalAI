# Saved models

The notebooks write reusable best-validation checkpoints here:

- `mpnn_qm9_<target>_best.pt`
- `deepaffinity_real.pt`
- `deepaffinity_synthetic.pt`
- `graphvae_qm9.pt`
- `JunctionTreeVAE/model.iter-*`
- `DeepDDS/deepdds_gat.pt`
- `MolGAN/molgan_qm9.pt`
- `HiGNN/hignn_<dataset>.pt`

Checkpoints include configuration and preprocessing metadata. By default, a
compatible checkpoint is loaded instead of retraining. Set `FORCE_RETRAIN =
True` in the relevant notebook to train and replace it.
