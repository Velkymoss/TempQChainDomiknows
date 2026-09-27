# Consistent Chronologies

This repository contains the source code, experiments, and evaluation pipeline for my master's thesis on using logical constraints for improving logical consistency of language models for temporal relation extraction. The primary objective of the thesis is to uncover the precise effects of integrating temporal rules via
differentiable soft logic during model training. Therefore, a BERT-base model was fine-tuned using a
primal-dual objective on the TimeBank-Dense (TB-Dense) corpus.

## Repository structure
``` text
.
├── data/     
│   ├── TimebankDense.full.txt
│   ├── tb_dense_dev.json
│   ├── tb_dense_test.json
│   ├── tb_dense_test_constraints.json
│   ├── tb_dense_train.json
│   └── timebank_1_2/      # TimeBank corpus files
│       ├── data/
│       │   ├── extra/     # Extra annotations. Used for create-data script
│       │   └── timeml/    # TimeML annotations
│       ├── docs/          
│       └── dtd/           
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks and analysis
│   ├── ablation.ipynb
│   └── loss.ipynb
├── scripts/               # Utility scripts
│   ├── calculate_story_tokensize.py
│   └── parse_output_files.py
├── runs/                  # output log files from slurm runs
├── src/
│   └── tempQchain/        # Main package
│       ├── __init__.py
│       ├── constraint_analysis.py
│       ├── data/          # Data processing
│       │   ├── __init__.py
│       │   ├── create_tb_dense.py
│       │   └── utils.py
│       ├── graphs/        # Constraint graphs
│       │   ├── __init__.py
│       │   └── graph.py
│       ├── logger.py
│       ├── main.py
│       ├── patches/       # Patched DomiKnows files
│       │   ├── __init__.py
│       │   ├── lossprogram.py
│       │   └── program.py
│       ├── programs/      # Program definitions
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── program.py
│       │   └── utils.py
│       ├── readers/       # Data readers
│       │   ├── __init__.py
│       │   ├── temporal_reader.py
│       │   └── utils.py
│       ├── train.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── graphs/           # Graph tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── graph.py
│   │   ├── run_tests.py
│   │   ├── test_inverse.py
│   │   ├── test_symmetric.py
│   │   └── test_transitive.py
│   └── readers/          # Reader tests
│       ├── conftest.py
│       └── test_temporal_reader.py
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock 
```
## Setup

### Prerequisites
- Python 3.11.11
- [uv](https://docs.astral.sh/uv/) package manager

### Dependencies
Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or alternatively via pip:
```bash
pip install uv
```
Then in the project root use:
```bash
uv venv
source .venv/bin/activate
uv sync
```
### Data
We are using a dense version of the [TimeBank corpus](https://aclanthology.org/P14-2082/). 

Create a `data` folder in the root of the project with following structure. :
- [TimebankDense.full.txt](https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.full.txt)
- `timebank_1_2` (folder with annotated TB-Dense articles, which are not freely available unfortunately)
- Note: The annotated articles from the `extra/` directory are used rather than those from `timeml/`.
For an exact overview please refer to the repository diagram.

Then run the create-data script to create the train/dev/test split for TB-dense:
```bash
tempqchain create-data --augment-train
```


## CLI usage

The CLI allows for fine-tuning a BERT-base model on TB-Dense using inverse and transitivity constraints:

### PMD constraint model
```bash
tempqchain train-model --pmd --use-class-weights --constraints --seed 0 --run-name some_run_name
```
### Baseline
For training a BERT baseline on the unaugmented dataset use:
```bash
tempqchain create-data --no-augment-train
tempqchain train-model --pmd --use-class-weights --no-constraints --seed 0 --run-name some_run_name
```
### Data augmented baseline
For training a Baseline without constraints on the augmented dataset use:
```bash
tempqchain create-data --augment-train
tempqchain train-model --pmd --use-class-weights --no-constraints --seed 0 --run-name some_run_name
```
### Constraint analysis
For inference of a fine-tuned model for constraint analysis and constraint satisfaction rates use:
```bash
tempqchain constraint-analysis --model name_of_saved_model --constraints --seed 0
```

## Experiment tracking

This project uses MLflow for experiment tracking.  
Enable MLflow with the `--use-mlflow` flag:
```bash
tempqchain train-model --use-mlflow
```
MLflow will log metrics, hyperparameters, and model artifacts.
By default, logs are saved locally in mlruns/. 
You can browse them with:
```bash
mlflow ui
```

## Reproducibility
All experiments use fixed random seeds 0,1, and 2. Note that GPU operations may still introduce small differences between runs.

## Tests
### Unit tests

```bash
uv run pytest
```
### Graph tests
Graph tests must run in isolation due to side effects.
```bash
uv run python tests/graphs/run_tests.py
```

## Patched files

Due to some limitations in the `domiknows==0.533` library, I decided to patch two library files.
The files can be found under `src/tempQchain/patches`. The paths of the files to patch are `domiknows/program/lossprogram.py` and `domiknows/program/program.py`. The made changes include a customized training interface and a fix for a file-name error produced by the library.

## Code provenance and contributions

This repository is a fork of [SpaRTUNQChain](https://github.com/HLR/SpaRTUNQChain), originally developed by Premsri and Kordjamshidi (2025) for neuro-symbolic spatial reasoning. Their approach integrates logical constraints into language model training using a primal-dual objective implemented with DomiKnowS. This project adapts their approach and codebase to temporal relation extraction.

Some parts were taken over without much adaptation, while others were heavily customized:
- The `TemporalReader` class was developed specifically for this project.
- The graph implementation was adapted to support the temporal constraints used in this project.
- The `program` implementation was adapted for temporal relation extraction.
- The BERT PyTorch model was heavily adapted for the task.
- The main training script was developed specifically for this project.

The JSON structure of our dataset directly stems from  Premsri and Kordjamshidi. The create-data script was primarily contributed by Vasiliki Kougia, who also co-supervised this thesis. I have modified the script at certain points, fixed bugs, and included it into the main pipeline.

## Thesis

**Title:** Consistent Chronologies: Evaluating Logical Constraints for Temporal Relation Extraction in Text

**Author:** Maximilian Moser  
**Aspired Degree:** MA Digital Humanities  
**University:** University of Vienna  
**Year:** 2026

The full thesis is available [here](thesis.pdf).

## Citation

If you use this repository or build upon this work, please cite:

``` bibtex
@mastersthesis{moser2026consistentchron,
  author = {Maximilian Moser},
  title  = {Consistent Chronologies: Evaluating Logical Constraints for Temporal Relation Extraction in Text},
  school = {University of Vienna},
  year   = {2026}
}
```
## Contact
For questions about this work:  
Maximilian Moser  
a11811692@unet.univie.ac.at

## Acknowledgment
This work builds upon the [DomiKnowS](https://aclanthology.org/2021.emnlp-demo.27/) framework developed by Rajaby Faghihi et al. (2021). I would like to thank the authors and contributors for developing and making the framework publicly available.

## Use of AI tools
AI tools were used for code-related research and programming assistance, including researching implementation approaches, looking up syntax and library usage, debugging, and code review.

## References

- Hossein Rajaby Faghihi, Quan Guo, Andrzej Uszok, Aliakbar Nafar, and Parisa Kordjamshidi. 2021. [*DomiKnowS: A Library for Integration of Symbolic Domain Knowledge in Deep Learning*](https://aclanthology.org/2021.emnlp-demo.27/). In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 231–241, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

- Tanawan Premsri and Parisa Kordjamshidi. 2025. [*Neuro-symbolic Training for Reasoning over Spatial Language*](https://aclanthology.org/2025.findings-naacl.128/). In *Findings of the Association for Computational Linguistics: NAACL 2025*, pages 2395–2414, Albuquerque, New Mexico. Association for Computational Linguistics.