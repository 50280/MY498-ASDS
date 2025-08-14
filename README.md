# Capstone Project, MSc in Applied Social Data Science 

The present repository contains the code and other materials required to replicate the capstone project titled 'A taxonomy of Interrogatives and Its Role in Human-Language Model Interactions'.
Description of the analyses are in the main report for this project, which is included in this repository under ...04_final_report.

## Primary Contact Information

-   **Name**: Anonymous Student
-   **Email address**: [a.student\@lse.ac.uk](mailto:a.student@lse.ac.uk)
-   **Phone**: +44 1234 567890
-   **Time Zone**: GMT

## How can future users replicate this project and associated analyses?

### 1. Clone the repository
After cloning the repository to a given directory, all file paths in the respective setup chunks of the scripts should be updated to this path. These are always in the setup chunks under a 'Directory management' subheader, and saved as 'wdir' objects in .Rmd files, and PROJECT_ROOT in .ipynb notebooks. 

### 2. Clone the python environment
This can be done via the terminal, running the code below

```bash
# After cloning or forking this repository and setting the terminal to it via cd
conda env create -f 02_code/00_setup-requirements/environment.yml
conda activate capstone_env
```
### 3. Download the data
Solely one dataset is not included in the present repository because of file size limitations. It it stored via HuggingFace and can directly be downloaded using the datasets library, as demonstrated in the code below. 

```python
from datasets import load_dataset
import os

# Set your project root
PROJECT_ROOT = "/path/to/cloned/folder/here"

# Define target folder
target_dir = os.path.join(PROJECT_ROOT, "01_data", "07_final_classified_data")
os.makedirs(target_dir, exist_ok=True)

# Download dataset
dataset = load_dataset("carowagner/study2")
```

**Prerequisites:** A variety of research methods were implemented in the present project. Accordingly, each script has a 'Setup' chunk at the very top, which describes what the prerequisites for the given script are, and what needs to be done to implement them. Some of these require individual API keys, for which secure storage options are described in the scripts. 


## Folder Structure

```{bash}
MY498-capstone-main/
├── 00_helper_documents/
│   └── interrogatives_taxonomy.pdf # pdf with schema of operationalisation
│      
├── 01_data/
│   ├── 01_PRISM_data/ # All data files from Kirk et al. (2024)
│   │   ├── survey.jsonl 
│   │   ├── utterances.jsonl
│   │   └──... 
│   ├── 02_data-labelling/ # All raw and individual labelled files
│   │   ├── 01_labelling-instructions/
│   │   │   ├── labeller_answer_guidelines.pdf
│   │   │   └── labeller_information_sheet.pdf
│   │   ├── 02_prompts-to-label/
│   │   │   ├── one_percent_prompts.csv
│   │   │   └── ten_percent_prompts.csv
│   │   ├── 03_main-labeller/ # All labelled files
│   │   ├── 04_labeller-1/
│   │   └── 07_round_two_labelling/
│   ├── 03_BERT_fine-tuning/
│   │   ├── Q1A_train.csv # One train and one validation file for each fine-tuned BERT
│   │   ├── Q1A_validation.csv
│   │   ├── Q1B_train.csv
│   │   └── ... 
│   ├── 05_utterance_classification/ # Intermediary data files
│   │   ├── PRISM_filtered.csv
│   │   └── PRISM_final_categories.csv
│   ├── 06_design-based_supervised-learning/
│   │   ├── labelled_gold_standard.csv
│   │   └── PRISM_classification_outcomes.csv
│   ├── 07_final_classified_data/
│   │   └── final_data_with_certainty.jsonl # Final dataset used for operationalisation and Study 1, can be downloaded on HuggingFace
│   ├── 08_descriptive_results/
│   │   ├── dsl_category_proportions.csv
│   │   ├── logit_summary_table.csv
│   │   ├── marginal_effects_table.csv
│   │   └── prediction_table.csv
│   ├── 09_experiment_implementation/
│   │   ├── 01_prompts_for_manipulation.csv
│   │   └── 02_model_responses/
│   │       ├── all_model_responses_combined.jsonl # File with all raw LLM responses combined
│   │       ├── anthropic_claude_*.jsonl # One file for each LLM type 
│   │       ├── deepseek_ai_*.jsonl
│   │       ├── google_gemini-*.jsonl
│   │       └── ...
│   └── 10_experimental_results/
│       ├── ate_and_level_effects_and_llm.csv
│       ├── marginal_effects_specification.csv
│       └── ...
│      
├── 02_code/
│   ├── 00_setup-requirements/
│   │   ├── BERT_fine_tune_args.py # Full BERT fine-tuning args
│   │   ├── environment.yml # File to reproduce the env
│   │   └── python_requirements.txt
│   ├── 01_helper-functions/
│   │   ├── 00_general_helper.R
│   │   ├── 01_helper_04_testing-models.R
│   │   ├── experimental_results_api_calls.py
│   │   └── general_python_helper.py
│   ├── 01_sample_data_subsets.Rmd
│   ├── 02_inter_coder_reliability.Rmd
│   ├── 03_classifying_questions.ipynb
│   ├── 04_design_based_supervised-learning.Rmd
│   ├── 05_estimate_uncertainty.ipynb
│   ├── 06_taxonomy_limitations.ipynb
│   ├── 07_descriptive_log_regs.Rmd
│   ├── 08_descriptive_analyses.ipynb
│   ├── 09_experiment_calculations.Rmd
│   ├── 10_prep_experiment_population.ipynb
│   ├── 11_implement_experiment.ipynb
│   ├── 12_experimental_analyses.Rmd
│   └── 13_format_tables.ipynb
│      
├── 03_outputs/
│   ├── 01_taxonomies_of_interrogatives/
│   └── ... # Each folder contains the outputs relevant to that section included in the final report
│   ├── 02_descriptive_analyses/
│   │   ├── figure_2.pdf
│   │   └──...
│   ├── 03_experimental_analyses/
│   │   └──...
│   └── 04_appendices/
│   │   └──...
│      
├── 04_final_report/
│   ├── apa.csl
│   ├── MY498_50280_report.pdf
│   ├── MY498_50280_report.Rmd
│   ├── MY498_50280_report.tex
│   └── references.bib
│      
└── README.md


```
