# Predicting Student Academic Risk: A Fairness-Aware Approach

## Abstract

Student dropout remains a persistent challenge in higher education, with significant implications for educational access and equity. This project develops and evaluates machine learning models to predict student academic outcomes (Dropout, Enrolled, Graduate) using a real dataset of 4,424 students from a Portuguese higher education institution. Beyond standard predictive modeling, this work emphasizes three dimensions often overlooked in similar projects: rigorous handling of class imbalance, model interpretability via SHAP, and a fairness audit across demographic and socioeconomic subgroups. A Random Forest classifier achieved the best performance (Macro F1 = 0.696), narrowly outperforming a tuned XGBoost model with SMOTE oversampling (Macro F1 = 0.685). Interpretability analysis identified second-semester course completion and tuition payment status as the two strongest predictors of dropout risk. The fairness audit revealed near-parity in performance across gender but a meaningful ~12-point Macro F1 gap disadvantaging scholarship-holding students, a finding with direct implications for responsible deployment of such models in real institutional settings.

## 1. Introduction

Student attrition is both a personal setback for affected students and a systemic inefficiency for institutions and society, particularly when it disproportionately affects students from lower-income or otherwise disadvantaged backgrounds. Early identification of at-risk students could enable timely intervention, but predictive models deployed in this space carry real ethical weight: a model that performs unevenly across student subgroups risks reinforcing the very inequities it aims to address.

This project asks two research questions:
1. **Predictive**: Can academic, demographic, and socioeconomic features reliably predict whether a student will drop out, remain enrolled, or graduate?
2. **Equity**: Does predictive performance hold consistently across demographic and socioeconomic subgroups, or does the model exhibit systematic blind spots that would matter for real-world deployment?

## 2. Related Work

Student dropout prediction has been studied extensively using both classical statistical models and modern machine learning. Tinto's foundational model of student departure emphasizes the role of early academic and social integration in determining persistence, a framework consistent with this project's finding that early-semester performance dominates predictive power. Realinho et al. (2021/2022), who originally compiled and published the dataset used here, applied several classical ML approaches and reported comparable class-imbalance challenges. More recent work in the fairness-in-ML literature (e.g., studies on algorithmic bias in educational and lending contexts) has emphasized that models optimized purely for aggregate accuracy can obscure meaningful performance disparities across subgroups — a concern this project directly investigates rather than assumes away.

*(Note: replace with your actual citations once you've pulled 2-3 real papers from Google Scholar — searching "student dropout prediction machine learning" and "algorithmic fairness education" will surface strong candidates.)*

## 3. Dataset

The dataset consists of 4,424 students enrolled across various undergraduate programs at a Portuguese higher education institution, with 36 features spanning demographic information (age, gender, nationality), socioeconomic indicators (parental occupation and education, scholarship status, tuition payment status), and academic performance metrics (admission grade, first- and second-semester course enrollment/approval/grades). The target variable has three classes with the following distribution:

| Class | Count | Percentage |
|---|---|---|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 17.9% |

No missing values were present in the dataset. The class imbalance — particularly the underrepresentation of the "Enrolled" class — motivated the imbalance-handling strategies described in Section 4.

## 4. Methodology

**Data splitting.** The dataset was split into training (70%, n=3,096), validation (15%, n=664), and test (15%, n=664) sets using stratified sampling to preserve class proportions across all splits, mitigating the risk of an unrepresentative evaluation split given the class imbalance.

**Preprocessing.** Numeric features were standardized using `StandardScaler`, fit exclusively on the training set and applied to validation/test sets to prevent data leakage. The categorical target variable was label-encoded (Dropout=0, Enrolled=1, Graduate=2).

**Models compared.**
- *Logistic Regression* (baseline floor), with `class_weight="balanced"` to counteract class imbalance
- *Random Forest* (200 trees), also class-weighted
- *XGBoost*, tuned via grid search over `max_depth` (4/6/8), `n_estimators` (100/200), and `learning_rate` (0.05/0.1), combined with SMOTE oversampling applied only to the training set

**Evaluation metric.** Macro-averaged F1 score was used as the primary metric rather than accuracy, since accuracy would be misleadingly dominated by the majority "Graduate" class given the imbalance.

**Interpretability.** SHAP (SHapley Additive exPlanations) values were computed using `TreeExplainer` on the best-performing tree-based model to identify which features most influence predictions toward the Dropout class.

**Fairness audit.** Model performance (Macro F1, and Dropout-class recall specifically) was evaluated separately across subgroups defined by Gender and Scholarship holder status, to assess whether the model performs equitably across these dimensions.

## 5. Results

### 5.1 Model Comparison

| Model | Macro F1 | Notes |
|---|---|---|
| Logistic Regression | 0.664 | Baseline; weakest overall, as expected of a linear model |
| Random Forest | **0.696** | Best-performing model |
| XGBoost (tuned + SMOTE) | 0.685 | Slightly underperformed Random Forest despite tuning and oversampling |

Notably, the additional complexity introduced by XGBoost's hyperparameter tuning and SMOTE's synthetic oversampling did not yield an improvement over the simpler Random Forest baseline. This suggests that Random Forest's bagging-based ensemble already captures the moderate class imbalance in this dataset reasonably well, and that synthetic oversampling — while theoretically appealing — did not provide additional generalizable signal for the minority "Enrolled" class in this case. All three models struggled comparatively with the "Enrolled" class, which is intuitive: a currently-enrolled student's eventual outcome is inherently more ambiguous than a student who has already dropped out or graduated.

### 5.2 Interpretability (SHAP)

The SHAP summary analysis (Figure: `outputs/shap_summary_dropout.png`) identified the following as the strongest predictors of Dropout risk, in order:

1. **Curricular units 2nd sem (approved)** — by far the dominant predictor; fewer approved courses strongly pushes predictions toward Dropout
2. **Tuition fees up to date** — a near-binary signal; students not current on fees are at sharply elevated dropout risk
3. **Curricular units 1st sem (approved)** — the same pattern as #1, slightly weaker, for the first semester
4. **Course** — dropout risk varies meaningfully by academic program
5. **Curricular units 2nd sem (enrolled)** — lower enrollment counts correlate with disengagement
6. **Mother's occupation** — a socioeconomic proxy with real predictive signal

These findings are consistent with Tinto's model of student departure, which emphasizes early academic integration as the strongest leading indicator of persistence. Notably, "Tuition fees up to date" surfacing as the second-strongest predictor directly exposes financial precarity as a driver of dropout, rather than requiring it to be inferred indirectly — a finding directly relevant to the fairness results discussed below.

### 5.3 Fairness Audit

| Subgroup | n | Macro F1 |
|---|---|---|
| Gender = 0 | 449 | 0.682 |
| Gender = 1 | 215 | 0.686 |
| No scholarship | 510 | 0.690 |
| Scholarship holder | 154 | **0.574** |

Performance across gender showed near-parity (a 0.4-point difference, within expected noise). However, a substantial gap emerged for scholarship status: the model performs 12 Macro-F1 points worse for scholarship-holding students than for non-recipients.

## 6. Discussion

The concentration of predictive power in early academic engagement metrics (course approvals, enrollment counts) confirms that dropout is, to a meaningful extent, observable well before it occurs — supporting the case for early-warning systems built on exactly these signals. The prominence of tuition payment status as a predictor further suggests that financial precarity is a direct, measurable risk factor rather than a hidden confound.

The fairness gap for scholarship holders warrants particular attention. Two plausible explanations exist, and they are not mutually exclusive: (1) scholarship holders represent a smaller subgroup (n=154 vs. n=510 in the validation set), giving the model comparatively less signal to learn their patterns during training; and (2) scholarship status may correlate with a more heterogeneous population — potentially mixing high-need students facing distinct risk factors with high-achieving merit recipients — whose outcomes are genuinely harder to predict from the available features. Distinguishing between these explanations would require either a larger sample of scholarship recipients or additional features capturing the basis of scholarship award (need-based vs. merit-based), neither of which is available in this dataset.

This finding has direct implications for responsible deployment: an institution using this model to prioritize dropout-prevention outreach should be explicitly aware that its predictions are less reliable for scholarship recipients — arguably a population where accurate early identification matters most, given their likely socioeconomic vulnerability. Three mitigation directions follow: (a) collecting additional training data specifically for this subgroup, (b) applying a subgroup-weighted training objective that penalizes errors on scholarship holders more heavily, or (c) at minimum, communicating this known limitation explicitly to any advisor or administrator using the model's output, so predictions for this subgroup are treated with appropriate caution rather than equal confidence.

## 7. Limitations

- The dataset originates from a single institution in a single country, limiting generalizability to other educational contexts and systems
- No temporal validation was performed across multiple academic years; the model's stability over time is unknown
- The class imbalance, while addressed via class weighting and SMOTE, was not fully resolved — the minority "Enrolled" class remained the hardest to predict across all models
- The fairness audit was limited to two subgroups (gender, scholarship status) due to available features; other equity-relevant dimensions (e.g., first-generation student status, rural/urban origin) were not present in the dataset and could not be evaluated
- SMOTE's synthetic oversampling, while standard practice, generates synthetic feature combinations that may not correspond to plausible real students, and its neutral-to-negative effect on final performance here (relative to Random Forest) suggests it should not be assumed beneficial by default

## 8. Conclusion

This project demonstrates that student academic outcomes can be predicted with reasonable accuracy from institutional data available early in a student's academic career, with second-semester course completion and tuition payment status emerging as the strongest actionable signals. Equally important, the fairness audit surfaced a genuine equity concern — reduced predictive reliability for scholarship-holding students — that would be invisible under standard aggregate accuracy reporting alone. Future work should prioritize closing this fairness gap before any such model is considered for real institutional deployment, and should investigate whether additional features capturing the basis and context of financial aid could help explain and reduce the observed disparity.