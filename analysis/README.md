# Model analysis

## Card input PCA

Render all 52 cards from the newest durable local checkpoint:

```bash
.venv/bin/python analysis/plot_card_embedding_pca.py
```

The six panels show every pair among PC1, PC2, PC3, and PC4 for the effective
card-dependent PLAY-event input (`card_emb + rank_emb + suit_emb`). Every panel
contains and labels all 52 cards.

Render the corresponding output embeddings—the 52 weight rows that map the
shared model state to card-action logits—with:

```bash
.venv/bin/python analysis/plot_card_embedding_pca.py --source action-head
```

This writes `analysis/card_action_head_pca.png`. The action-head's per-card
scalar biases are not included in PCA; the plotted vectors are the learned
directions in state space used to score each card.

## Complementary action-head geometry

Install the analysis dependencies once:

```bash
uv sync --group analysis
```

Generate four distinct PNGs from the latest checkpoint:

```bash
.venv/bin/python analysis/plot_card_action_head_geometry.py
```

The outputs are:

- `card_action_head_mds.png`: metric MDS over cosine distances.
- `card_action_head_umap.png`: cosine UMAP with a fixed seed.
- `card_action_head_cosine_heatmap.png`: side-by-side cosine similarity in
  fixed suit-major and rank-major order.
- `card_action_head_probes.png`: cross-validated suit and rank linear probes
  compared with shuffled-label distributions.

The suit probe holds out one complete rank at a time, while the rank probe
holds out one complete suit at a time. This tests whether each property
generalizes across the other property instead of memorizing individual cards.

Run the same four analyses for the effective input card representation
(`card_emb + rank_emb + suit_emb`) with:

```bash
.venv/bin/python analysis/plot_card_action_head_geometry.py --source input
```

These are written as `card_input_embedding_mds.png`,
`card_input_embedding_umap.png`, `card_input_embedding_cosine_heatmap.png`, and
`card_input_embedding_probes.png`.

Choose a checkpoint or output explicitly when needed:

```bash
.venv/bin/python analysis/plot_card_embedding_pca.py \
  --checkpoint checkpoints/local/v9_8m_laptop_seed1/plump_v4_iter_14018.pt \
  --output analysis/card_embedding_pca.png
```
