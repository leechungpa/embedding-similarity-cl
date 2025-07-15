<h1 align="center"> <p>On the Similarities of Embeddings in Contrastive Learning</p></h1>
<h3 align="center">
    <p>Chungpa Lee, Sehee Lim, Kibok Lee, Jy‑yong Sohn</p>
    <p>Yonsei University</p>
    <!-- <a href="https://">
        <img src="https://img.shields.io/badge/paper-blue.svg" alt="paper">
    </a> -->
    <a href="https://arxiv.org/abs/2506.09781">
        <img src="https://img.shields.io/badge/arXiv-b31b1b.svg" alt="arXiv">
    </a>
</h3>

Contrastive learning operates on a simple yet effective principle: Embeddings of positive pairs are pulled together, while those of negative pairs are pushed apart. In this paper, we propose a unified framework for understanding contrastive learning through the lens of cosine similarity, and present two key theoretical insights derived from this framework. First, in full-batch settings, we show that perfect alignment of positive pairs is *unattainable* when negative-pair similarities fall below a threshold, and this misalignment can be mitigated by incorporating within-view negative pairs into the objective. Second, in mini-batch settings, smaller batch sizes induce stronger separation among negative pairs in the embedding space, i.e., *higher variance* in their similarities, which in turn degrades the quality of learned representations compared to full-batch settings. To address this, we propose an auxiliary loss that reduces the variance of negative-pair similarities in mini-batch settings. Empirical results show that incorporating the proposed loss improves performance in small-batch settings.


## Experiments

### Pretraining

```bash
# Pretraining on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-name pretrain_cifar.yaml

# Pretraining on ImageNet
CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py \
    --config-name pretrain_imagenet.yaml
```

The main implementation of the proposed loss is as follows:

```python
z1 = F.normalize(z1, dim=1)  # Normalized embedding vectors with shape (batch size, embedding dimension)
z2 = F.normalize(z2, dim=1)  # Normalized embedding vectors with shape (batch size, embedding dimension)
similarity_matrix = torch.einsum("id, jd -> ij", z1, z2)

index = index.unsqueeze(0)  # Index of instances in the batch; the same index denotes positive pairs
neg_mask = index.t() != index

loss += (similarity[neg_mask] + 1/n).pow(2).mean()
```

This loss can be enabled through the configuration file as follows:

```bash
add_vrn_loss_term:
  enabled: True
  weight: # The hyperparameter lambda
  k: # The number of training instances
```

### Linear Probing

```bash
# Evaluation on CIFAR
CUDA_VISIBLE_DEVICES=0 python3 main_linear.py \
    --config-name linear_cifar.yaml

# Evaluation on ImageNet
CUDA_VISIBLE_DEVICES=0 python3 main_linear.py \
    --config-name linear_imagenet.yaml
```


## Citation

```tex
@InProceedings{lee2025similarities,
  title={On the Similarities of Embeddings in Contrastive Learning},
  author={Lee, Chungpa and Lim, Sehee and Lee, Kibok and Sohn, Jy-yong},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```