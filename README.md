# Sentiment Analysis on Restaurant Reviews

## Project Overview
This project aims to classify restaurant reviews as **positive** or **negative** using a sentiment analysis model. By analyzing customer feedback, restaurants can gain valuable insights to improve their services and enhance customer satisfaction.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Dependencies](#dependencies)
- [Implementation](#implementation)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project consists of restaurant reviews with the following features:
- **Review Text**: The content of the review.
- **Liked**: A binary label indicating whether the review is positive (`1`) or negative (`0`).

## Features
The sentiment analysis model extracts relevant features from the text, such as:
- Text preprocessing (e.g., removing special characters, stopwords).
- Tokenization.
- Vectorization using techniques like Bag-of-Words or TF-IDF.

## Dependencies
The project is implemented in Python, and the following libraries are required:

```bash
pip install pandas numpy matplotlib scikit-learn nltk
```

- **Pandas**: Data preprocessing and exploration.
- **NumPy**: Numerical operations.
- **Matplotlib**: Visualization of data.
- **Scikit-learn**: Machine learning models and evaluation.
- **NLTK**: Natural Language Processing tasks (e.g., tokenization, stopword removal).

## Implementation
The project follows these steps:

1. **Data Loading and Exploration**
   - Load the dataset using `pandas`.
   - Explore dataset properties (e.g., check for null or duplicate values).

2. **Data Preprocessing**
   - Clean and preprocess text data by removing noise.
   - Tokenize the reviews and remove stopwords using NLTK.
   - Convert text data into numerical form using vectorization techniques.

3. **Model Training**
   - Split the dataset into training and testing sets.
   - Train a classification model using the Naive Bayes algorithm (specifically Multinomial Naive Bayes).

4. **Model Evaluation**
   - Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
   - Visualize results using confusion matrices.

## Results
The Naive Bayes model achieved the following:
- **Accuracy**: 80%
- **Precision**: 80% (macro average)
- **Recall**: 80% (macro average)
- **F1-Score**: 80% (macro average)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-restaurant-reviews.git
   cd sentiment-analysis-restaurant-reviews
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script:
   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

4. Use the model to classify your own reviews:
   - Provide input reviews.
   - Obtain predictions indicating whether the review is positive or negative.

## Future Work
- Implement deep learning techniques (e.g., LSTM, BERT) for improved accuracy.
- Expand the dataset to include reviews from diverse sources.
- Develop a web application for user-friendly predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for suggestions and improvements.

## License
This project is licensed under the [MIT License](LICENSE).

