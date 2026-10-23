# Fake News Classification with NLP

An English-language news classification project combining exploratory data analysis, text preprocessing, Bag of Words features and a comparison of **Multinomial Naive Bayes** and **MLPClassifier**.

Developed around the **Veridian Analytics** project scenario, with particular attention to whether classifiers learn editorial-source patterns rather than signals that generalise to new articles.

**[View the analysis notebook](fake_news_classification.ipynb.ipynb)**

The notebook and its commentary are in Italian. This README provides an English overview.

## Project objective

Build and evaluate a prototype that assigns English articles to the dataset labels **`TRUE`** and **`FAKE`**, then export the preprocessing configuration and trained classifier for potential downstream integration.

The repository contains the analysis notebook, not a working Chrome extension. The classification output is not independent fact-checking or proof of an article's truthfulness.

## Dataset

The notebook downloads the [dataset archive supplied for the project](https://proai-datasets.s3.eu-west-3.amazonaws.com/fake_news.zip), containing `Fake.csv` and `True.csv`.

| Stage | Articles |
|---|---:|
| Original FAKE collection | 23,481 |
| Original TRUE collection | 21,417 |
| Combined corpus | 44,898 |
| Duplicate copies removed using title and body | 5,793 |
| Corpus after deduplication | 39,105 |
| Training set | 31,284 |
| Test set | 7,821 |

Available fields are `title`, `text`, `subject` and `date`. The model input combines the title and article body. Articles with an empty body but a non-empty title are retained.

Exact duplicates are removed before an **80/20 stratified split** with `random_state=42`. The `subject` field is excluded from model features because its categories separate the source collections and their labels.

## Workflow

1. **Explore the corpus:** inspect missing values, duplicates, class balance, subject distributions, title length, capitalisation, punctuation and common words.
2. **Investigate source bias:** examine Reuters references and other editorial or technical markers associated with the labels.
3. **Preprocess text:** lowercase, remove punctuation and numbers, lemmatise with spaCy's `en_core_web_sm`, remove stopwords and normalise remaining characters and whitespace.
4. **Vectorise:** fit `CountVectorizer(max_features=10000)` on the training texts only, then transform the test texts with the same vocabulary.
5. **Compare models:** train Multinomial Naive Bayes and a one-hidden-layer neural network, then inspect classification reports, confusion matrices and errors.
6. **Run a marker-removal experiment:** remove selected source-related tokens, fit a new training vocabulary and retrain both models using the same split.
7. **Export the selected pipeline:** save the classifier, vectorizer, preprocessing configuration and a Python prediction module, then reload them for a functional check.

The MLP uses 100 hidden units, logistic activation, the Adam solver, `max_iter=100`, `tol=0.005` and `random_state=42`.

## Recorded results

The following results are taken from the notebook's saved test-set outputs; they are not a new training run performed for this README.

| Model | Text configuration | Accuracy | F1 — FAKE | F1 — TRUE |
|---|---|---:|---:|---:|
| Multinomial Naive Bayes | Standard cleaned text | 94.76% | 0.9433 | 0.9512 |
| MLPClassifier | Standard cleaned text | 98.94% | 0.9884 | 0.9902 |
| Multinomial Naive Bayes | Selected source markers removed | 93.57% | 0.9311 | 0.9397 |
| **MLPClassifier — selected model** | **Selected source markers removed** | **98.41%** | **0.9826** | **0.9854** |

The final model makes **124 errors** on 7,821 test articles:

- 74 articles labelled FAKE are predicted TRUE.
- 50 articles labelled TRUE are predicted FAKE.

The notebook also demonstrates predictions on two synthetic texts. These examples test pipeline functionality and do not provide independent performance evidence.

## Source-bias experiment

The removed tokens are:

```text
reuters, reuter, getty, https, flickr, youtu,
tmsnrt, cdata, filessupport, quot
```

The selected MLP retains high accuracy after their removal, but this does **not** establish that source bias has been eliminated. Writing style, topic distribution and collection practices can still distinguish the two archives.

## Interpretation and limitations

- **Same-corpus evaluation:** the test set comes from the same collections as the training data. Performance on unseen publishers, different periods or other domains is not established.
- **Development-set reuse:** the same test split is used to compare models and preprocessing variants. A separate untouched evaluation set would provide a stronger final assessment.
- **Incomplete bias removal:** deleting a small marker list does not remove all source, stylistic or topic-related signals.
- **Duplicate coverage:** exact title/body duplicates are removed, but near-duplicate articles and related stories are not explicitly grouped across the split.
- **Confidence is not truth:** error analysis includes highly confident incorrect predictions. Returned probabilities are not validated measures of factual reliability.
- **Prototype scope:** external-source evaluation, probability calibration and ongoing monitoring would be needed before public-facing deployment.

## Repository contents and generated files

The repository currently contains [fake_news_classification.ipynb.ipynb](fake_news_classification.ipynb.ipynb) and this README. Analysis tables and charts are embedded in the notebook.

Running the export cells generates **`/content/fake_news_model_export.zip`**, containing:

| Generated file | Purpose |
|---|---|
| `fake_news_classifier.pkl` | Final MLP, fitted vectorizer, source-marker list and configuration |
| `fake_news_predictor.py` | Loading, preprocessing and prediction functions |
| `requirements.txt` | Versions of NumPy, SciPy, scikit-learn and spaCy from the export environment |
| `README.txt` | Instructions for using the exported classifier |

These are generated runtime artifacts and are **not currently committed to the repository**. The notebook's saved outputs document a reload-and-predict check after export.

## How to run

The notebook is designed for **Google Colab**, using `/content` paths and `google.colab.files` for the final download.

1. Download the notebook and upload it to Colab.
2. Ensure the dependencies and English spaCy model are available. If needed, add and run this setup cell before the analysis:

```python
%pip install pandas numpy scipy matplotlib scikit-learn spacy
!python -m spacy download en_core_web_sm
```

3. Run the notebook cells from top to bottom with Internet access for the dataset download.
4. Allow time for text preprocessing and training; execution time depends on the runtime.
5. Run the export cells to create and download the ZIP package.

The initial download cell replaces any existing `/content/fake_news_data` extraction folder. For local Jupyter execution, adapt the `/content` paths and the Colab-only download cell. Library and language-model versions may affect reproduced results.

## Tools and skills

Python · Pandas · NumPy · Matplotlib · spaCy · scikit-learn · Bag of Words · Text classification · Model comparison · Source-bias analysis · Error analysis · Model serialisation

## Author

[Claudio Lombardi](https://github.com/PsyClaudio)
