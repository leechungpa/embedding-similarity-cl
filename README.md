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

Contrastive learning (CL) operates on a simple yet effective principle: embeddings of positive pairs are pulled together, while those of negative pairs are pushed apart. Although various forms of contrastive loss have been proposed and analyzed from different perspectives, prior works lack a comprehensive framework that systematically explains a broad class of these objectives. In this paper, we present a unified framework for understanding CL, which is based on analyzing the cosine similarity between embeddings. In full-batch settings, we show that perfect alignment of positive pairs is unattainable when similarities of negative pairs fall below a certain threshold, and that this misalignment can be alleviated by incorporating within-view negative pairs. In mini-batch settings, we demonstrate that smaller batch sizes incur stronger separation among negative pairs within batches, which leads to higher variance in similarities of negative pairs. To address this limitation of mini-batch CL, we introduce an auxiliary loss term that reduces the variance of similarities of negative pairs in CL. Empirical results demonstrate that incorporating the proposed loss consistently improves the performance of CL methods in small-batch training.

## Codes

We are currently organizing the code and preparing the documentation. The full implementation will be uploaded **by the end of June 2025**.

## Citation

```tex
@InProceedings{lee2025similarities,
  title={On the Similarities of Embeddings in Contrastive Learning},
  author={Lee, Chungpa and Lim, Sehee and Lee, Kibok and Sohn, Jy-yong},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
