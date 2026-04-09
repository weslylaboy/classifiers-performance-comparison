# Bank Marketing Campaign: Comparing Classifiers

## Goal of the Project
The goal of this project is to help a Portuguese bank improve its telemarketing campaigns. I want to predict if a client will say "yes" or "no" to a term deposit after being contacted by phone. By knowing who is more likely to sign up, the bank can save time and money by calling the right people.

We compared four different models:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Trees
4. Support Vector Machines (SVM)

## Notebook Link
[Jupyter Notebook](prompt_III.ipynb)

## Understanding the Data
The dataset comes from 17 marketing campaigns between 2008 and 2010. It has over 41,000 records of clients. Each record includes:
- **Client info:** Age, job, marriage status, education, etc.
- **Campaign info:** How they were contacted, how long the call lasted, and past results.
- **Social/Economic info:** Interest rates and employment trends.

## Data Cleaning and Preparation
To get the data ready for our models, we did the following steps:
1. **Removed the 'default' column:** Most of the data was 'no' or 'unknown', so it didn't help our models.
2. **Kept 'unknown' as a category:** For columns like job and education, we kept 'unknown' as its own group to see if it mattered.
3. **Filtered for Bank Features:** As requested, we focused on 6 main features: `age`, `job`, `marital`, `education`, `housing`, and `loan`.
4. **Encoding and Scaling:** 
   - **Categorical Columns:** `job`, `marital`, `education`, `housing`, and `loan` were changed into numbers using **One-Hot Encoding**.
   - **Numerical Column:** `age` was adjusted using **StandardScaler** so its large numbers wouldn't confuse the models.
5. **Target Variable:** The goal (`y`) was changed to `1` for "yes" and `0` for "no".

## What I found in Data Analysis (EDA)
- **Big Imbalance:** Only about 11% of the people said "yes". This is a big problem because a model could just guess "no" every time and be 89% accurate but totally useless.
- **Jobs Matter:** Students and retired people were more likely to say "yes" compared to blue-collar workers.
- **Economic Factors:** We saw that interest rates and employment rates were very linked. When the economy is stable, people might be more willing to save.

## Models Before Fixing the Imbalance
At first, we ran the models with their default settings. They were not good because they almost always predicted "no".

| Model | Train Accuracy | Test Accuracy | ROC-AUC | Recall (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| Dummy (Baseline) | 88.76% | 88.65% | 0.50 | 0.00% |
| Logistic Regression | 88.76% | 88.65% | 0.50 | 0.00% |
| KNN | 90.38% | 87.36% | 0.52 | 8.34% |
| Decision Tree | 91.20% | 86.68% | 0.53 | 16.58% |
| SVC | 88.83% | 88.67% | 0.50 | 0.11% |

**Why they were not good:** Even though the accuracy looks high (~88%), the **Recall** was almost 0. This means the models were "lazy" and just guessed "no" for everyone. They couldn't find the actual subscribers.

## Improving the Models with GridSearchCV and SMOTE
To fix the "lazy" models, we used **SMOTE** (which creates fake examples of "yes" cases to balance the data) and **GridSearchCV** to find the best settings for each model.

### Best Parameters Found:
- **Logistic Regression:** `C=0.1`, `penalty='l1'`, `solver='liblinear'`. The 'l1' penalty helped pick the best features.
- **KNN:** `n_neighbors=51`. A larger neighborhood helped smooth out the noise.
- **Decision Tree:** `max_depth=5`, `min_samples_split=30`. Keeping the tree short stopped it from memorizing the data.
- **SVC:** `C=0.1`, `class_weight='balanced'`, `gamma='scale'`. This made the model more aggressive in finding "yes" cases.

## Final Model Results
After fixing the balance, the models became much more useful. They found the subscribers.

| Model | Test Accuracy | ROC-AUC | Recall (Class 1) |
| :--- | :--- | :--- | :--- |
| **SVC (Best Performance)** | **62%** | **0.6308** | **58%** |
| **Logistic Regression (Best Choice)** | **62%** | **0.6282** | **53%** |
| KNN | 65% | 0.6015 | 48% |
| Decision Tree | 70% | 0.5931 | 40% |

**Explanation of Metrics:**
- **Recall (53-58%):** This is the most important part. It means we are now catching more than half of all people who would say "yes".
- **ROC-AUC (0.63):** This shows how well the model separates "yes" from "no". 0.5 is guessing, so 0.63 is a good improvement given we only used 6 demographic features.

## Conclusion and Recommendations for the Bank
1. **Use Logistic Regression:** Even though SVC was slightly better at finding "yes" cases, it took 47 minutes to train. Logistic Regression took only 10 seconds and gave almost the same result. It is the best tool for daily use.
2. **Quality over Quantity:** By using these models, the bank can skip calling 60% of people who would have said "no". This saves a lot of time for the sales team.
3. **Personalized Messages:** Since students and retired people are more likely to subscribe, the bank should create special messages just for them.
4. **Next Steps:** To get even better results, the bank should start tracking how long the calls last and use that info in the next model, as call duration is a very strong hint of interest.


