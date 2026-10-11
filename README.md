# RefGAP

Training-free correction of reference-key attention logits for diffusion-based visual editing (Wang et al., https://arxiv.org/abs/2609.35708).

Install with `pip install -e .`. Run tests with `python -m pytest`.

This package is the logit correction and the layer-selection rule only. The appendix floor at 1e-6 and the edit-bias clip to [0, 8] are applied. The remaining mismatch is no diffusion editor and no images.
