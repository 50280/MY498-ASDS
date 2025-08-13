
# Requirements
library(tidyverse)
library(stringr)
library(caret)

################################################################################
# (1) Function that cleans main labeller datasets.
################################################################################

clean_and_standardize_labels <- function(df) {
  df %>%
    rename_with(~ str_replace_all(., "^X", "Q")) %>%
    mutate(across(where(is.character), ~ str_squish(.))) %>%
    mutate(across(where(is.character), ~ case_when(
      . == "y" ~ "YES",
      . == "n" ~ "NO",
      TRUE ~ .
    ))) %>%
    mutate(across(where(is.character), ~ replace_na(., "M")))
}


################################################################################
# (2) Function that adds true columns to labels. 
################################################################################

add_true_labels <- function(question_col, predicted_df, labelled_df) {
  true_col <- paste0(question_col, "_true")
  
  result_df <- predicted_df %>%
    mutate(
      !!question_col := replace_na(.data[[question_col]], "M")  # <-- fix here
    ) %>%
    left_join(
      labelled_df %>% select(utterance_id, !!sym(question_col)),
      by = "utterance_id",
      suffix = c("", "_true")
    ) %>%
    rename(
      !!true_col := !!sym(true_col)
    ) %>%
    mutate(
      !!true_col := replace_na(.data[[true_col]], "M")
    )
  
  return(result_df)
}


################################################################################
# (3) Function to compute F1 score. 
################################################################################

compute_weighted_f1 <- function(question_col, df) {
  true_col <- paste0(question_col, "_true")
  
  pred <- factor(df[[question_col]])
  true <- factor(df[[true_col]])
  
  all_classes <- union(levels(pred), levels(true))
  pred <- factor(pred, levels = all_classes)
  true <- factor(true, levels = all_classes)
  
  class_counts <- table(true)
  total <- sum(class_counts)
  
  f1_scores <- sapply(all_classes, function(class) {
    cm <- confusionMatrix(pred, true, positive = class)
    precision <- cm$byClass["Pos Pred Value"]
    recall <- cm$byClass["Sensitivity"]
    
    if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
      return(0)
    }
    
    2 * (precision * recall) / (precision + recall)
  })
  
  weights <- class_counts[all_classes] / total
  weighted_f1 <- sum(f1_scores * weights, na.rm = TRUE)
  
  return(weighted_f1)
}

