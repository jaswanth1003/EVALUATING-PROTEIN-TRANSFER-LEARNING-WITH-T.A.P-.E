# EVALUATING-PROTEIN-TRANSFER-LEARNING-WITH-T.A.P-.E

# Evaluating Protein Transfer Learning with TAPE

This project reproduces and evaluates models and experiments from the [TAPE: Tasks Assessing Protein Embeddings](https://arxiv.org/abs/1906.08230) paper.  
TAPE benchmarks the effectiveness of deep learning methods, especially transfer learning, on a variety of protein sequence tasks.

---

##  Paper Details

- **Paper Title**: TAPE: Tasks Assessing Protein Embeddings
- **Authors**: Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Peter Chen, John Canny, Pieter Abbeel, Yun Song
- **Published**: NeurIPS 2019
- **Link**: [https://arxiv.org/abs/1906.08230](https://arxiv.org/abs/1906.08230)

---

##  Project Goals

- Pre-train and fine-tune transformer-based models on protein sequences.
- Evaluate transfer learning performance on key downstream protein tasks.
- Reproduce benchmark results to validate model performance.
- Analyze and interpret the effectiveness of learned protein representations.

---

##  Project Structure

| Folder | Description |
|:-------|:------------|
| `data/` | Downloaded or generated datasets for various protein tasks |
| `models/` | Transformer architectures for protein modeling |
| `scripts/` | Pretraining, finetuning, and evaluation scripts |
| `results/` | Checkpoints, logs, and evaluation outputs |
| `configs/` | YAML config files for experiments |

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jaswanth1003/EVALUATING-PROTEIN-TRANSFER-LEARNING-WITH-T.A.P-.E.git
cd EVALUATING-PROTEIN-TRANSFER-LEARNING-WITH-T.A.P-.E

pip install -r requirements.txt


## Important Notes

Model checkpoints (*.pt files) larger than 100MB are NOT included directly in GitHub due to file size restrictions.

Please download or generate checkpoints locally.

Large files should be managed with Git LFS if necessary.

## Acknowledgements

The original TAPE codebase provided the backbone for much of the task implementations.

Thanks to the authors of TAPE for releasing the datasets and models!




