# Smart Education System Using AI and Blockchain

## Project overview
An experimental smart education framework combining Artificial Intelligence
and Blockchain technology for student performance prediction, learning
recommendations, and academic-record integrity verification.

## Features
- Educational CSV dataset upload and validation
- Dataset cleaning and overview
- Random Forest, Decision Tree, and Logistic Regression
- Student performance classification
- Rule-based personalized learning recommendations
- SHA-256 hash-linked local record chain
- Blockchain integrity verification
- Accuracy, precision, recall, and F1-score comparison

## Technologies
Python, Streamlit, Pandas, NumPy, scikit-learn, SHA-256.

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Dataset
The application accepts educational CSV datasets. The UCI Student Performance
dataset is a suitable starting dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

For the UCI dataset, choose `G3` as the target and exclude `G1` and `G2`
to avoid using earlier-period grades as predictors of the final grade.

## Research limitations
The blockchain component is a local SHA-256 hash-linked chain. It is not a
distributed blockchain and does not provide consensus or independent nodes.
Recommendations are transparent rule-based suggestions. Predictions shown
for the uploaded dataset are illustrative; held-out test metrics should be
used to report model performance.
