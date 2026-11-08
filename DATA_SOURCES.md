# Dataset source, license, and attribution

## Bank Account Fraud (BAF), Base variant

- **Creators:** Sérgio Jesus, José Pombal, Duarte Alves, André Cruz, Pedro Saleiro, Rita Ribeiro, João Gama, and Pedro Bizarro.
- **Dataset source:** [Kaggle Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), version 2, `Base.csv` only.
- **Official documentation:** [Feedzai BAF repository](https://github.com/feedzai/bank-account-fraud) and [NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html).
- **Dataset license shown on the Kaggle card:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Description:** Synthetic, privacy-preserving application-fraud data generated from an anonymized real-world bank-account-opening dataset. The Base variant has one million instances, application features, a month field, and fraud labels.
- **Reproducibility fingerprint:** SHA-256 for the local source CSV used in the recorded run: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`.

The project does not include or redistribute the raw dataset, fitted model weights, or row-level prediction outputs. Users must review the dataset's current license and terms before downloading, reusing, or adapting BAF. This public repository shares aggregate benchmark findings for non-commercial research and education, with attribution; where those findings qualify as adaptations of the dataset, the dataset's share-alike conditions apply. This project does not grant commercial rights to BAF or its adaptations.

The repository's MIT license applies to original source code only. It does not relicense the BAF dataset or override its terms for materials derived from it. The dataset's license is separate from any license applying to Feedzai's code.

## Citation

Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R., Gama, J., & Bizarro, P. (2022). “Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.” *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track*. [Paper page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html). DOI: [10.52202/068431-2432](https://doi.org/10.52202/068431-2432).

## Scope

This benchmark uses synthetic data only. It is not evidence of performance on a live bank, real customers, or real-world fraud patterns. The evaluation is descriptive for one chronological split and makes no production or fraud-reduction claim.
