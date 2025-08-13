

################################################################################
# Functions used across several scripts 
################################################################################

# (1) Function to read and clean JSONL file
read_clean_jsonl <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)
  lines <- gsub("\\bNaN\\b", "null", lines)  # Replace NaN with null
  json_data <- lapply(lines, fromJSON)  # Convert each line to JSON
  return(do.call(rbind, json_data))  # Combine into a data frame
}


# (2) Function to unnest selected nested columns in json files
unnest_columns <- function(df, cols) {
  for (col in cols) {
    if (col %in% names(df)) {
      # (a) Flatten the nested column
      expanded <- jsonlite::flatten(df[[col]])
      
      # (b) Prefix column names
      colnames(expanded) <- paste0(col, "_", colnames(expanded))
      
      # (c) Bind to main df
      df <- df %>%
        select(-all_of(col)) %>%
        bind_cols(expanded)
    }
  }
  return(df)
}


# (3)Function to refactor covariates for to ensure same data-preprocessing for implementation
# of DSL and analyses that are part of the project. 
refactor_covariates <- function(df) {
  df %>%
    mutate(
      gender = relevel(factor(gender), ref = "Male"),
      age_group = relevel(factor(age), ref = "18-24 years old"),
      birth_region = relevel(factor(location_birth_region), ref = "Europe"),
      religion_main = relevel(factor(religion_simplified), ref = "No Affiliation"),
      ethnicity_main = relevel(factor(ethnicity_simplified), ref = "White"),
      conversation_type = relevel(factor(conversation_type), ref = "unguided"),
      education_recode = relevel(factor(education_recode), ref = "Primary &/or Secondary Education"),
      participant_id = factor(user_id)
    )
}


# (4) Function to ensure aligned factor levels when using the model.matrix function
# and creating the g random forest model to implement the design-based supervised learning 
# estimator.
harmonize_factor_levels <- function(covariates) {
  for (var in covariates) {
    all_levels <- union(levels(factor(gstan_one_percent[[var]])),
                        levels(factor(prism_final_categories[[var]])))
    gstan_one_percent[[var]] <<- factor(gstan_one_percent[[var]], levels = all_levels)
    prism_final_categories[[var]] <<- factor(prism_final_categories[[var]], levels = all_levels)
  }
}


################################################################################
# Function for 02_inter_coder_reliability.rmd
################################################################################

# (1) Function that finds incongruences between labelers. 
find_differences_between <- function(dataset1, dataset2, question_name) {
  
  diffs <- dataset1 %>%
    select(utterance_id, user_prompt, !!sym(question_name)) %>%
    rename(
      user_prompt_1 = user_prompt,
      answer_1 = !!sym(question_name)
    ) %>%
    inner_join(
      dataset2 %>%
        select(utterance_id, user_prompt, !!sym(question_name)) %>%
        rename(
          user_prompt_2 = user_prompt,
          answer_2 = !!sym(question_name)
        ),
      by = "utterance_id"
    ) %>%
    filter(answer_1 != answer_2)
  
  return(diffs)
}

# (2) Function to compute Krippendorf's alpha
compute_krippendorff_alpha <- function(df) {
  ratings_matrix <- as.matrix(df %>% select(rater1, rater2, rater3))
  storage.mode(ratings_matrix) <- "character" 
  ratings_matrix <- t(ratings_matrix) # make sure that the matrix is transposed           
  irr::kripp.alpha(ratings_matrix, method = "nominal")
}


# (3) More fine-grained functions for understanding differences.
# (should have reused previous functions here)
compute_alpha_labmain_labtwo <- function(q) {
  df <- labmain_labtwo_paired %>%
    select(!!sym(paste0(q, "_labmain")), !!sym(paste0(q, "_lab2"))) %>%
    rename(rater1 = 1, rater2 = 2)
  
  ratings_matrix <- as.matrix(df)
  storage.mode(ratings_matrix) <- "character"
  ratings_matrix <- t(ratings_matrix)
  
  irr::kripp.alpha(ratings_matrix, method = "nominal")
  
  
  compute_alpha_two_raters <- function(q) {
    df <- labmain_labone_paired %>%
      select(!!sym(paste0(q, "_labmain")), !!sym(paste0(q, "_labone"))) %>%
      rename(rater1 = 1, rater2 = 2)
    
    ratings_matrix <- as.matrix(df)
    storage.mode(ratings_matrix) <- "character"
    ratings_matrix <- t(ratings_matrix)
    
    irr::kripp.alpha(ratings_matrix, method = "nominal")
  }
}



################################################################################
# Compute contingency tables using the DSL package - used in study 1 
################################################################################

# Description of what the function below does from the final report: 
# This uses the DSL package, regressing each dummy-encoded interrogative category i on 
# each demographic feature d using linear regression without an intercept. This approach is 
# mathematically equivalent to computing subgroup means. Although it is less interpretable, 
# this was necessary because the DSL estimator cannot simply be applied to aggregated count data.

#  (1) DSL Estimation Function for mean proportions. 
estimate_dsl_proportions <- function(pred_col, label_col, demo_var) {
  # (a) Compute sampling probability from labeled data
  n_labeled <- sum(!is.na(df[[label_col]]))
  n_total <- nrow(df)
  sampling_prob <- n_labeled / n_total
  
  # (b) Prepare DSL input
  dsl_df <- df %>%
    select(all_of(c(demo_var, label_col, pred_col))) %>%
    mutate(
      group = .[[demo_var]],
      Y = .[[label_col]],
      pred_Y = .[[pred_col]],
      sample_prob = ifelse(!is.na(Y), sampling_prob, NA)
    ) %>%
    select(group, Y, pred_Y, sample_prob) %>%
    filter(!is.na(group) & !is.na(pred_Y))
  
  # (c) Skip if no usable data
  if (nrow(dsl_df) == 0) {
    message(glue("Skipping: No data for {pred_col}, {label_col}, {demo_var}"))
    return(tibble())
  }
  
  # (d) Fit DSL model
  formula_str <- as.formula("Y ~ as.factor(group) - 1")
  model <- dsl(
    model = "lm",
    formula = formula_str,
    predicted_var = "Y",
    prediction = "pred_Y",
    data = dsl_df
  )
  
  # (e) Extract and check coefficients
  summary_out <- summary(model)
  
  # (f) Convert to dataframe and assign names
  result_df <- as.data.frame(summary_out)
  if (nrow(result_df) == 0) {
    message(glue("Skipping: Empty model output for {pred_col}, {label_col}, {demo_var}"))
    return(tibble())
  }
  
  # (g) Assign expected column names (if missing or unnamed)
  colnames(result_df) <- c("Estimate", "Std.Error", "CI.Lower", "CI.Upper", "p.value", "Signif")
  
  # (h) Format output
  result_df <- result_df %>%
    rownames_to_column("group") %>%
    mutate(
      group = gsub("as.factor\\(group\\)", "", group),
      estimate_exp = exp(Estimate),
      conf.low_exp = exp(CI.Lower),
      conf.high_exp = exp(CI.Upper),
      adj_p = p.adjust(p.value, method = "bonferroni"),
      question_type = pred_col,
      label_type = label_col,
      demographic = demo_var
    )
  
  return(result_df)
}

