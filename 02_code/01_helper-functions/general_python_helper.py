
################## Overview ##################

# Aim: This file contians helper functions that are used 
# throughout the entire research project. 
# Date of last modification: 06.08.2025


##############################################

# Imports
import re
import tiktoken
import pandas as pd
import numpy as np
from scipy.stats import levene
from scipy.stats import ttest_rel
from collections import Counter
import statsmodels.formula.api as smf
from sklearn.metrics import classification_report


##############################################
# General
##############################################

# Define function to assign interrogative category types 
def assign_question_types_taxonomy(row):
    """Function to implement the logical conditions that are part of the
    operationalisation of the taxonomy of interrogatives stemming from 
    Belnap & Steel (1976) using the outputs from the seven fine-tuned 
    BERT classifiers."""
    matched = {
        'M': 0,
        'why_q': 0,
        'whether_q': 0,
        'which_q': 0,
        'whathow_q': 0,
        'hobsons_c': 0
    }

    # (1) Not an interrogative - excluded from downstream analyses
    if row['label_1A'] == 'n' and row['label_1B'] == 'n':
        matched['M'] = 1
        return pd.Series(matched)

    # (2) Why
    if row['label_3A'] == 'y' and row['label_2C'] == '1':
        matched['why_q'] = 1
        return pd.Series(matched)

    # (3) Whether
    if (
        (row['label_2A'] == 'y' and row['label_2C'] == '2') or
        (row['label_2B'] == 'y' and str(row['label_2C']).isdigit() and int(row['label_2C']) > 1)
    ):
        matched['whether_q'] = 1
        return pd.Series(matched)

    # (4) Which
    if row['label_2C'] == 'u' and row['label_4A'] in ['o', 'n']:
        matched['which_q'] = 1
        return pd.Series(matched)

    # (5) What/How
    if row['label_4A'] == 'y' and row['label_2C'] == 'u':
        matched['whathow_q'] = 1
        return pd.Series(matched)

    # In terms of the logic, it is important that this one be implemented
    # at the end, as otherwise, with the way the operationalsiation of this
    # specific category is currently setup, there would be overlaps with the other
    # interrogative categories, and it would not be mutually exclusive with the
    # other categories. (because of OR operator)
    # (6) Hobson’s Choice
    # This automatically also assigns a 1 to hobsons_c if both 1B and 3A have YES. 
    if row['label_1B'] == 'y' or row['label_3A'] == 'y':
        matched['hobsons_c'] = 1
        return pd.Series(matched)

    return pd.Series(matched)


##############################################
# For file: 06_taxonomy_performance_limitations.py
##############################################

def classify_mc_dropout(row):
    """Runs over outputs ffrom monte-carlo simulation with dropout for 
    uncertainty, and classifies the different outputs using the logic operationalised
    through the taxonomy of interrogatives. 
    Inputs: Row (pd.Series) with MC predictions for each classifier.
    Returns: dict, count of classifications per category, including 'not_matched'."""

    # Extract class labels
    labels = {
        k.split('_')[-1]: row[f'mc_data_class_{k.split("_")[-1]}']
        for k in row.keys() if k.startswith('mc_data_class_')
    }

    final_counts = Counter()

    for i in range(100):
        # Build a row for the ith iteration
        label_at_i = {k: labels[k][i] for k in labels}
        row_i = pd.Series({f'label_{k}': v for k, v in label_at_i.items()})

        # Classify using taxonomy
        result = assign_question_types_taxonomy(row_i)

        # Extract matched category
        matched_categories = [k for k, v in result.items() if v == 1]

        if not matched_categories:
            final_counts['not_matched'] += 1  # new label for unmatched
        else:
            label = matched_categories[0].replace('_q', '').replace('_c', '')
            final_counts[label] += 1

    return dict(final_counts)


################################################################################
# Functions to compute and format the entropy values for the final uncertainty estimations. 
################################################################################

def get_mc_mode(count_dict):
    """Function to return the category with the highest count from an input dictionary."""
    return max(count_dict.items(), key=lambda x: x[1])[0]


def compute_mc_entropy(count_dict):
    """Computes entropy of a Monte Carlo prediction distribution.
    Takes a dictionary of category counts and returns the 
    entropy as a measure of uncertainty in the classification."""

    total = sum(count_dict.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in count_dict.values()])
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def compute_entropy_summary(df, category_col='mc_final_interrogative_category', entropy_col='mc_entropy'):
    """Compute entropy summary statistics grouped by the predicted category.
    Inputs:
        df (pd.DataFrame): The input dataframe.
        category_col (str): Column name for predicted category.
        entropy_col (str): Column name for entropy values.
    Outputs: pd.DataFrame: A summary table with per-category entropy 
        stats and an 'ALL' row.
    """
    # Per-category summary
    group_summary = (
        df.groupby(category_col)[entropy_col]
        .agg(['mean', 'std', 'median', 'min', 'max'])
        .rename(columns={
            'mean': 'mean_entropy',
            'std': 'std_entropy',
            'median': 'median_entropy',
            'min': 'min_entropy',
            'max': 'max_entropy'
        })
    )

    #  Get overall stats
    overall_stats = df[entropy_col].agg(['mean', 'std', 'median', 'min', 'max'])
    overall_stats.index = [
        'mean_entropy', 'std_entropy', 'median_entropy', 'min_entropy', 'max_entropy'
    ]

    # Append overall to the bottom
    group_summary.loc['ALL'] = overall_stats

    return group_summary


##############################################
# Functions to evaluate classification performance
##############################################

def evaluate_classifiers(merged_df, classifier_map):
    """
    Evaluate multiple fine-tuned classifiers using classification_report.
    Parameters:
    - merged_df: DataFrame containing gold and predicted labels, already merged on utterance_id
    - classifier_map: dict mapping classifier names (e.g., "1A") to original column names (e.g., "Q1A")
    Returns:
    - overall_perf_df: DataFrame with overall accuracy, weighted F1, and support per classifier
    - category_perf_df: DataFrame with per-class accuracy (recall), F1, and support
    """
    
    overall_metrics = []
    category_metrics = []

    for k in classifier_map:
        gold_col = f"label_{k}_gold"
        pred_col = f"label_{k}_bert"

        if gold_col not in merged_df.columns or pred_col not in merged_df.columns:
            print(f"Missing columns for {k}, skipping.")
            continue

        data = merged_df[[gold_col, pred_col]].dropna()
        if data.empty:
            print(f"No data for {k}, skipping.")
            continue

        y_true = data[gold_col]
        y_pred = data[pred_col]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Overall metrics
        accuracy = report["accuracy"]
        weighted_f1 = report["weighted avg"]["f1-score"]
        total_support = sum(
            v["support"] for label, v in report.items()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        )
        overall_metrics.append({
            "classifier": k,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "support": total_support
        })

        # Per-class metrics
        for label, scores in report.items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                continue
            category_metrics.append({
                "classifier": k,
                "category": label,
                "accuracy": scores["recall"],  # recall used as class-level accuracy
                "f1": scores["f1-score"],
                "support": scores["support"]
            })

    overall_perf_df = pd.DataFrame(overall_metrics)
    category_perf_df = pd.DataFrame(category_metrics)

    return overall_perf_df, category_perf_df


def evaluate_question_types(df, categories):
    """Evaluates classification performance for each category 
    by comparing predicted and gold labels. Returns a DataFrame with 
    accuracy, precision, recall, F1-score, and support for class '1'."""
    results = []

    for cat in categories:
        y_true = df[f"{cat}_gold"]
        y_pred = df[f"{cat}_bert"]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cls1 = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        accuracy = (y_true == y_pred).mean()  # manual accuracy

        results.append({
            "category": cat,
            "accuracy": accuracy,
            "precision": cls1["precision"],
            "recall": cls1["recall"],
            "f1": cls1["f1-score"],
            "support": cls1["support"]
        })

    return pd.DataFrame(results)


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Parameters:
        y_true: array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_prob: array-like of shape (n_samples,)
            Predicted probabilities for the positive class.
        n_bins: int
            Number of bins to split probabilities into.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for b in range(n_bins):
        bin_mask = bin_ids == b
        if np.any(bin_mask):
            acc = np.mean(y_true[bin_mask] == (y_prob[bin_mask] >= 0.5))
            conf = np.mean(y_prob[bin_mask])
            ece += np.abs(acc - conf) * np.sum(bin_mask) / len(y_true)
    return ece

################################################################################
# Function to compute over-representation factors for descriptive analyses
################################################################################

def compute_orf(df, demo_col, qtype_cols, clean_qtype_labels):
    """
    Computes over-representation factors (ORFs) for each subgroup within a demographic feature,
    using design-adjusted interrogative scores as weighted outcomes.
    """
    # Compute baseline proportions for each group in the demographic column
    if isinstance(df[demo_col].dtype, pd.CategoricalDtype):
        df[demo_col] = df[demo_col].cat.set_categories(df[demo_col].cat.categories, ordered=True)
        base_props = df[demo_col].value_counts(normalize=True).reindex(df[demo_col].cat.categories)
    else:
        base_props = df[demo_col].value_counts(normalize=True).sort_index()

    # Prepare ORF results table
    orf_table = pd.DataFrame(index=base_props.index)

    for q in qtype_cols:
        weights = df[q]  # design-adjusted probability values
        N_q = weights.sum()

        if N_q == 0 or base_props.isna().all():
            orf_table[clean_qtype_labels[q]] = np.nan
            continue

        # Sum weights per group (use observed=False to retain full category structure)
        dq_weights = df.groupby(demo_col, observed=False)[q].sum().sort_index()
        dq_props = dq_weights / N_q  # observed group share
        orf = dq_props / base_props  # over-representation factor

        orf_table[clean_qtype_labels[q]] = orf

    return orf_table


################################################################################
# Study 2. Function to count all response lengths with the same tokeniser
################################################################################

def count_tokens(text, encoding):
    """Function to count the number of tokens from an LLM response. 
    Takes text string as input and uses the tiktoken library.
    Takes string argument for the type of encoding wanted, e.g., 'gpt-4'."""
    if pd.isna(text):
        return np.nan
    return len(encoding.encode(text))



################################################################################
# Study 2. Function to format df for paired token length comparison
################################################################################

def create_paired_token_dataframe(df, group_cols=['model_provider', 'model_name', 'utterance_id'], 
                                  column='model_fed_q_type', value='tokens',
                                  prompt_types=('hobsons_c', 'whathow_q')):
    """
    Creates a pivoted dataframe with token counts for both prompt types, paired by LLM and utterance.
    Parameters:
    df (pd.DataFrame): Input dataset
    group_cols (list): Columns to group by for pairing
    column (str): Column indicating prompt type
    value (str): Column containing the value (e.g., token counts)
    prompt_types (tuple): Pair of prompt types to retain (must match values in `column`)
    Returns:
        pd.DataFrame: Pivoted dataframe with paired token counts
    """
    pivot_df = (
        df
        .pivot_table(
            index=group_cols,
            columns=column,
            values=value
        )
        .dropna(subset=list(prompt_types))  # keep only paired entries
        .reset_index()
    )
    return pivot_df


################################################################################
# Study 2. Function to compare token lenghts, does means (SDs), paired t-tests & cohen's d
################################################################################


def compare_token_lengths_by_prompt_type(df):
    """
    Compare average token lengths between 'whathow_q' and 'hobsons_c' prompt types,
    grouped by model_provider and model_name. Performs paired t-tests and calculates Cohen's d.
    Params:
        df (pd.DataFrame): DataFrame containing model outputs with token counts.
    Returns:
        pd.DataFrame: Summary statistics including means, SDs, t-test results, df, and Cohen's d.
    """

    # (1) Pivot the data to get paired token counts per utterance
    pivot_df = create_paired_token_dataframe(df)

    # (2) Initialize list to collect summary stats
    summary_stats = []

    # (3) Loop through each LLM (by provider and model name)
    for (provider, model), group in pivot_df.groupby(['model_provider', 'model_name']):
        hobsons_tokens = group['hobsons_c']
        whathow_tokens = group['whathow_q']
        
        # (a) Paired t-test
        t_stat, p_val = ttest_rel(whathow_tokens, hobsons_tokens)
        df_ttest = len(whathow_tokens) - 1  # Degrees of freedom
        
        # (b) Cohen’s d
        diff = whathow_tokens - hobsons_tokens
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan

        # (c) Append stats
        summary_stats.append({
            'LLM provider': provider,
            'model_name': model,
            'n_whathow': len(whathow_tokens),
            'mean_whathow_tokens': round(whathow_tokens.mean(), 2),
            'std_whathow_tokens': round(whathow_tokens.std(ddof=1), 2),
            'n_hobsons': len(hobsons_tokens),
            'mean_hobsons_tokens': round(hobsons_tokens.mean(), 2),
            'std_hobsons_tokens': round(hobsons_tokens.std(ddof=1), 2),
            't_value': round(t_stat, 2),
            'p_value': round(p_val, 4),
            'df': df_ttest,
            'cohens_d': round(cohens_d, 3)
        })

    return pd.DataFrame(summary_stats)

################################################################################
# Study 2. Function to get the prompt that the LLM was fed for the qualitative analyses 
################################################################################

# Function to get the actual prompt text fed to the LLM
def get_fed_prompt(row):
    """
    Given the fed question type ('hobsons_c' or 'whathow_q'), return the actual prompt text
    the LLM responded to by selecting the correct column.
    """
    # Determine which column the LLM response is based on
    if row['model_fed_q_type'] == 'hobsons_c':

        # If LLM was fed a Hobson's Choice prompt, get the text that was classified as 'hobsons_c'
        return row['user_prompt'] if row['hobsons_c_debiased'] == 1 else row['counterfactual_prompt']
    elif row['model_fed_q_type'] == 'whathow_q':
        
        # If LLM was fed a What/How prompt, get the one not classified as 'hobsons_c'
        return row['counterfactual_prompt'] if row['hobsons_c_debiased'] == 1 else row['user_prompt']
    else:
        return None

################################################################################
# Study 2. Function to compute ATE and Levene's Test
################################################################################

def compute_treatment_effect_stats(data, attr, group_var="model_fed_q_type", group_levels=("hobsons_c", "whathow_q")):
    """
    Computes difference in means (ATE) and Levene's test for a given attribute and dataset.
    Params; data: pd.DataFrame - The dataset to use
    attr: str - The attribute column to analyze
    group_var: str - The column indicating group/treatment
    group_levels: tuple - The two levels to compare 
    Returns: dict with results
    """

    # Check if both groups exist
    if data[group_var].nunique() < 2:
        return None
    
    data = data.dropna(subset=[attr])

    # OLS for diff in means
    model = smf.ols(f"{attr} ~ C({group_var})", data=data).fit()
    term = f"C({group_var})[T.{group_levels[1]}]"

    coef = model.params.get(term, float('nan'))
    se = model.bse.get(term, float('nan'))
    pval = model.pvalues.get(term, float('nan'))
    ci = model.conf_int().loc[term].tolist() if term in model.params else [float('nan'), float('nan')]

    # Levenes test
    group1 = data[data[group_var] == group_levels[0]][attr]
    group2 = data[data[group_var] == group_levels[1]][attr]
    levene_stat, levene_p = levene(group1, group2, center='median')

    return {
        "attribute": attr,
        "ate": coef,
        "se": se,
        "p_value": pval,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "levene_stat": levene_stat,
        "levene_p_value": levene_p
    }

################################################################################
# Study 2. Functions for figure 3 - mostly to help formatting
################################################################################

# Function to clean the LLM provider names
def clean_model_name(name):
    """Takes in the different LLM names and returns a cleaned version
    that ensures proper formatting throughout."""
    name = name.replace("-", " ").strip().lower()
    
    if "gpt" in name:
        name = name.replace("gpt", "GPT")
    if name.startswith("meta llama"):
        name = name.replace("meta ", "")
    if name.startswith("claude"):
        name = re.sub(r"\s\d{6,}", "", name).strip()
    
    mistral_map = {
        "magistral medium 2506": "Magistral medium",
        "mistral medium latest": "Mistral medium",
        "mistral small": "Mistral small"
    }
    if name in mistral_map:
        name = mistral_map[name]
    
    return name.capitalize()

# Function to determine average marginal effects with high uncertainties.
def flag_high_uncertainty(group):
    """Function to determine which average marginal effects have values
    that I have defined to be of high uncertainty. These are defined as 
    confidence interval widths exceeding 1.5× the interquartile range 
    above the upper quartile for each attribute"""

    q1 = group["ci_width"].quantile(0.25)
    q3 = group["ci_width"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    return group["ci_width"] > upper_bound
