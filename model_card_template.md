# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained on Census Bureau data to predict whether an individual's income exceeds $50,000 per year.

## Intended Use
The model is intended for research purposes to understand the factors influencing income levels based on Census data. 

## Training Data
The training data is derived from the Census Bureau dataset, specifically the census.csv file.
It includes various demographic and socioeconomic features such as age, workclass, education, marital status, occupation, relationship, race, sex, and native country. 

## Evaluation Data
The evaluation data is a held-out portion of the Census Bureau dataset, typically 20% of the original data, used as a test set.

## Metrics
The model's performance on these metrics is as follows:
Precision: 0.7319
Recall: 0.6384
F1 Score: 0.6863

## Ethical Considerations
The model uses sensitive demographic information, including race and sex, which could lead to biased predictions if not carefully monitored.

## Caveats and Recommendations
Regular monitoring of the model's performance across different demographic groups is crucial to detect and mitigate potential biases.
The model should be retrained periodically with more recent data to ensure its relevance and accuracy.