# =============================================================================
# 1. Remove all objects from the workspace
# =============================================================================
rm(list = ls())

# =============================================================================
# 2. Load libraries
# =============================================================================
library(dplyr)
library(lme4)
library(broom.mixed)  # optional, for tidy model output
library(performance)  # for R² etc.
library(lmerTest)     # for p-values (optional)
library(corrplot)    # for correlation matrix visualization
library(ggplot2)     # for grid plot histograms
library(tidyr)       # for pivot_longer
library(reshape2)  # for melt()
library(caret)    # for createFolds
library(ggplot2)

# Seed
set.seed(2025)

# =============================================================================
# 3. Get Data
# =============================================================================
data_dir<-"directory"
setwd(data_dir)

df <- read.csv("input/pwi_merged_2025-06-24.csv", stringsAsFactors = FALSE)
csln_measures_german <- read.csv("input/pwi_csln_measures_german_fastText.csv", stringsAsFactors = FALSE)
csln_measures_english <- read.csv("input/pwi_csln_measures_english_fastText.csv", stringsAsFactors = FALSE)

# delete target and context column from csln_measures
csln_measures_german <- csln_measures_german[, -c(2, 3)]
csln_measures_english <- csln_measures_english[, -c(2, 3)]

# replace "_q" with "_q0"
colnames(csln_measures_german) <- gsub("_q", "_q0", colnames(csln_measures_german))
colnames(csln_measures_german) <- gsub("_q010", "_q10", colnames(csln_measures_german))
colnames(csln_measures_english) <- gsub("_q", "_q0", colnames(csln_measures_english))
colnames(csln_measures_english) <- gsub("_q010", "_q10", colnames(csln_measures_english))



# drop 

# delete rows where trial is NA
df <- df[!is.na(df$trial), ]

# =============================================================================
# 4. Define predictors 
# =============================================================================
model0_predictors <- c(
  # experiment
  
  "trial",
  "familiarization",
  "SOA",
  "further_tasks",
  "overt_naming",
  "collection_online",
  "experimental_language_german",
  
  # target
  "target_length",
  "target_zipf",
  "repetition_target",
  
  # distractor
  "context_length",
  "context_zipf",
  "repetition_context",
  "stimulus_is_text"
  
)

model1_predictors<-c(model0_predictors,"cosine_similarity")
model2_predictors<-c(model1_predictors,colnames(csln_measures_german)[2:11])
model3_predictors<-c(model2_predictors,colnames(csln_measures_german)[12:length(colnames(csln_measures_german))])
model3_predictors <- model3_predictors[!grepl("q01", model3_predictors)]



# =============================================================================
# 4. Get Data
# =============================================================================

# combine target and context
df$target_context<-paste0(df$target, "_", df$context)

head(df)
summary(df[model0_predictors])


############################# 
## G E R M A N    D A T A. ##
#############################

# german data /english data
df_german <- df[df$experimental_language_german == "1", ]

# left_join df and csln_measures
df_german <- left_join(df_german, csln_measures_german, by = c("target_context" = "target_distractor"))

# print colnames
colnames(df_german)

# delete all rows where p1_td is NA
df_german <- df_german[!is.na(df_german$p1_td), ]

# select only semantic studies
df_german <- df_german[df_german$study %in% c(
  "abdel_rahman_unpublished_1_eeg",
  "damian_2014",
  "jescheniak_2024b",
  "jescheniak_2024",
  "jescheniak_2020",
  "kuhlen_2022",
  "lorenz_2021",
  "lorenz_unpublished2_control",
  "vogt_2022"
), ] 

###############################
## E N G L I S H    D A T A. ##
###############################

# english data
df_english <- df[df$experimental_language_german == "-1", ]

# left_join df and csln_measures
df_english <- left_join(df_english, csln_measures_english, by = c("target_context" = "target_distractor"))

# print colnames
colnames(df_english)

# delete all rows where p1_td is NA
df_english <- df_english[!is.na(df_english$p1_td), ]

df_english <- df_english[df_english$study %in% c(
  "vieth_2014a",
  "cutting_1999",
  "gauvin_2018"
), ] 


###############################
## C O M B I N E    D A T A. ##
###############################
df_model<-rbind(df_english, df_german)


# =============================================================================
# 5. Scale Model 2 and Model 3 predictors
# =============================================================================

# Identify which columns to scale (all CSLN features only, not baseline predictors)
model_0_and_1_features_to_be_scaled<-c("SOA","trial","target_length", "context_length", "repetition_target","repetition_context","target_zipf", "context_zipf","cosine_similarity")
model2_cSLN_features <- colnames(csln_measures_german)[2:11]
model3_cSLN_features <- colnames(csln_measures_german)[12:length(colnames(csln_measures_german))]

# Scale Model 0 and Model 1 predictors
df_model[model_0_and_1_features_to_be_scaled] <- scale(df_model[model_0_and_1_features_to_be_scaled])

# Scale Model 2 predictors
df_model[model2_cSLN_features] <- scale(df_model[model2_cSLN_features])

# Scale Model 3 predictors
df_model[model3_cSLN_features] <- scale(df_model[model3_cSLN_features])



# =============================================================================
# 5. Correlation Matrix
# =============================================================================


# Correlation matrix for Model 3 features
mcor2<-cor(df_model[,model2_cSLN_features], use = "pairwise.complete.obs")
mcor3<-cor(df_model[,model3_cSLN_features], use = "pairwise.complete.obs")

numeric_column<-c()
for (i in 1:ncol(df_model)) {
    numeric_column[i]<-is.numeric(df_model[,i])
}

# correlation matrix for german data
cor_german<-cor(df_german[numeric_column], use = "pairwise.complete.obs")
sort(cor_german[,3])

# correlation matrix for english data
cor_english<-cor(df_english[numeric_column], use = "pairwise.complete.obs")
sort(cor_english[,3])



# =============================================================================
# 6. Define random effects
# =============================================================================
random_effects <- "(1 | participant) + (1 | experiment) "

# =============================================================================
# 7. Construct formulas for each model
# =============================================================================
model0_formula <- as.formula(
  paste("log_rt ~", paste(model0_predictors, collapse = " + "), "+", random_effects)
)

model1_formula <- as.formula(
  paste("log_rt ~", paste(model1_predictors, collapse = " + "), "+", random_effects)
)

model2_formula <- as.formula(
  paste("log_rt ~", paste(model2_predictors, collapse = " + "), "+", random_effects)
)

model3_formula <- as.formula(
  paste("log_rt ~", paste(model3_predictors, collapse = " + "), "+", random_effects)
)


# =============================================================================
# 8. Fit the models
# =============================================================================
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

cat("Fitting Model 0...\n")
model0 <- lmer(model0_formula, data = df_model, REML = FALSE)

cat("Fitting Model 1...\n")
model1 <- lmer(model1_formula, data = df_model, REML = FALSE)

cat("Fitting Model 2...\n")
model2 <- lmer(model2_formula, data = df_model, REML = FALSE)

cat("Fitting Model 3...\n")
model3 <- lmer(model3_formula, data = df_model, REML = FALSE)

# =============================================================================
# 9. Compare models using likelihood ratio tests
# =============================================================================
print("++++++++++----------------------------------++++++++++")
print("Model comparison using likelihood ratio tests")
print("++++++++++----------------------------------++++++++++")
anova_results <- anova(model0, model1, model2, model3)
print(anova_results)

# =============================================================================
# 10. Summary stats for each model
# =============================================================================
model_list <- list(
  Model0 = model0,
  Model1 = model1,
  Model2 = model2,
  Model3 = model3
)

print("Summary for each model")
for (name in names(model_list)) {
  cat("\n", name, ":\n")
  print(summary(model_list[[name]]))
}


# collect R² values for all models programmatically
r2_df <- lapply(names(model_list), function(name) {
  r2_values <- r2(model_list[[name]])
  data.frame(
    model = name,
    Conditional = r2_values$R2_conditional,
    Marginal = r2_values$R2_marginal
  )
}) %>%
  bind_rows()

# reshape for ggplot
r2_long <- r2_df %>%
  pivot_longer(cols = c(Marginal, Conditional),
               names_to = "R2_type",
               values_to = "R2_value")

# 2x1 Grid plot for R²
ggplot(r2_long, aes(x = model, y = R2_value, group = 1)) +
  geom_line(size = 1, color="#00376c") +
  geom_point(size = 3, color="#00376c") +
  facet_wrap(~ R2_type, nrow = 1, scales = "free_y", 
             labeller = labeller(R2_type = c(Conditional = "Conditional R²", 
                                            Marginal = "Marginal R²"))) +
  labs(
    y = "R² Value",
    x = "Model"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1, "cm")
  )

## ggsave pdf
#ggsave("output/plots/r2_models.pdf", width = 10, height = 4, dpi = 300)



####################################
## C R O S S  V A L I D A T I O N ##
####################################

# list of model formulas
model_formula_list <- list(
  Model0 = model0_formula,
  Model1 = model1_formula,
  Model2 = model2_formula,
  Model3 = model3_formula
)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

run_cv <- function(data, grouping_var, k = 5) {
  folds <- groupKFold(data[[grouping_var]], k = k)
  results <- data.frame()
  
  for (i in seq_along(folds)) {
    test_idx  <- folds[[i]]
    train_idx <- setdiff(seq_len(nrow(data)), test_idx)
    train_data <- data[train_idx, , drop = FALSE]
    test_data  <- data[test_idx, , drop = FALSE]
    
    for (mname in names(model_formula_list)) {
      formula <- model_formula_list[[mname]]
      
      fit <- try(lmer(formula, data = train_data, REML = FALSE, control = ctrl),
                 silent = TRUE)
      if (inherits(fit, "try-error")) {
        results <- rbind(results, data.frame(
          fold = i, model = mname, RMSE = NA_real_, R2 = NA_real_
        ))
        next
      }
      
      preds <- predict(fit, newdata = test_data, allow.new.levels = TRUE)
      ok <- is.finite(test_data$log_rt) & is.finite(preds)
      
      if (any(ok)) {
        rmse <- sqrt(mean((test_data$log_rt[ok] - preds[ok])^2))
        ss_res <- sum((test_data$log_rt[ok] - preds[ok])^2)
        ss_tot <- sum((test_data$log_rt[ok] - mean(test_data$log_rt[ok]))^2)
        r2 <- 1 - ss_res/ss_tot
      } else {
        rmse <- NA_real_
        r2 <- NA_real_
      }
      
      results <- rbind(results, data.frame(
        fold = i, model = mname, RMSE = rmse, R2 = r2
      ))
    }
  }
  
  summary <- results %>%
    group_by(model) %>%
    summarise(
      mean_RMSE = mean(RMSE, na.rm = TRUE),
      sd_RMSE   = sd(RMSE, na.rm = TRUE),
      mean_R2   = mean(R2, na.rm = TRUE),
      sd_R2     = sd(R2, na.rm = TRUE)
    ) %>%
    mutate(cv_type = grouping_var)
  
  return(summary)
}

# 1. Held-out participants
cv_participants <- run_cv(df_german, "participant", k = 5)

# 2. Held-out experiments
cv_experiments  <- run_cv(df_german, "experiment", k = 5)

# 3. Held-out trials (random)
trial_folds <- createFolds(1:nrow(df_german), k = 5)
cv_trials_results <- data.frame()
for (i in seq_along(trial_folds)) {
  test_idx  <- trial_folds[[i]]
  train_idx <- setdiff(seq_len(nrow(df_german)), test_idx)
  train_data <- df_german[train_idx, ]
  test_data  <- df_german[test_idx, ]
  
  for (mname in names(model_formula_list)) {
    formula <- model_formula_list[[mname]]
    fit <- try(lmer(formula, data = train_data, REML = FALSE, control = ctrl),
               silent = TRUE)
    if (inherits(fit, "try-error")) {
      cv_trials_results <- rbind(cv_trials_results,
                                 data.frame(fold = i, model = mname, RMSE = NA_real_, R2 = NA_real_))
      next
    }
    preds <- predict(fit, newdata = test_data, allow.new.levels = TRUE)
    ok <- is.finite(test_data$log_rt) & is.finite(preds)
    
    if (any(ok)) {
      rmse <- sqrt(mean((test_data$log_rt[ok] - preds[ok])^2))
      ss_res <- sum((test_data$log_rt[ok] - preds[ok])^2)
      ss_tot <- sum((test_data$log_rt[ok] - mean(test_data$log_rt[ok]))^2)
      r2 <- 1 - ss_res/ss_tot
    } else {
      rmse <- NA_real_
      r2 <- NA_real_
    }
    
    cv_trials_results <- rbind(cv_trials_results,
                               data.frame(fold = i, model = mname, RMSE = rmse, R2 = r2))
  }
}
cv_trials <- cv_trials_results %>%
  group_by(model) %>%
  summarise(mean_RMSE = mean(RMSE, na.rm = TRUE),
            sd_RMSE = sd(RMSE, na.rm = TRUE),
            mean_R2 = mean(R2, na.rm = TRUE),
            sd_R2 = sd(R2, na.rm = TRUE)) %>%
  mutate(cv_type = "trial")

# Combine summaries
cv_summary_all <- bind_rows(cv_participants, cv_experiments, cv_trials)
print(cv_summary_all)


ggplot(cv_summary_all, aes(x = model, y = mean_RMSE, color = cv_type, group = cv_type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(
    x = "Model",
    y = "Mean RMSE",
    color = "CV Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )





