import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

class Model:
    # Logistic Regression model with Word + Character TF-IDF features
    
    def __init__(self, C=0.1, solver='lbfgs', class_weight='balanced'):
        # Word-level TF-IDF
        self.tfidf_word = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.7,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # Character-level TF-IDF (captures slang/misspellings)
        self.tfidf_char = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=3000,
            min_df=10,
            max_df=0.9,
            sublinear_tf=True
        )
        
        self.model = LogisticRegression(
            C=C,
            solver=solver,
            class_weight=class_weight,
            max_iter=5000,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.metadata_features = []
    
    def extract_features(self, df, fit=False):
        reviews = df['review'].fillna('').values
        
        # Word TF-IDF features
        if fit:
            X_word = self.tfidf_word.fit_transform(reviews)
            X_char = self.tfidf_char.fit_transform(reviews)
        else:
            X_word = self.tfidf_word.transform(reviews)
            X_char = self.tfidf_char.transform(reviews)
        
        # Metadata features (log-scaled per Olmedilla et al. 2024)
        metadata = {}
        
        if 'review' in df.columns:
            metadata['log_review_length'] = np.log1p(df['review'].str.len().fillna(0).values)
        
        if 'author_playtime_at_review' in df.columns:
            metadata['log_playtime_at_review'] = np.log1p(df['author_playtime_at_review'].fillna(0).values / 60)
        
        if 'author_playtime_forever' in df.columns:
            metadata['log_playtime_forever'] = np.log1p(df['author_playtime_forever'].fillna(0).values / 60)
        
        if 'author_num_games_owned' in df.columns:
            metadata['log_games_owned'] = np.log1p(df['author_num_games_owned'].fillna(0).values)
        
        if 'author_num_reviews' in df.columns:
            metadata['log_num_reviews'] = np.log1p(df['author_num_reviews'].fillna(0).values)
        
        if 'steam_purchase' in df.columns:
            metadata['steam_purchase'] = df['steam_purchase'].fillna(True).astype(float).values
        
        if 'received_for_free' in df.columns:
            metadata['received_for_free'] = df['received_for_free'].fillna(False).astype(float).values
        
        if 'written_during_early_access' in df.columns:
            metadata['early_access'] = df['written_during_early_access'].fillna(False).astype(float).values
        
        if fit:
            self.metadata_features = list(metadata.keys())
        
        # Scale metadata
        metadata_arr = np.column_stack(list(metadata.values())) if metadata else np.array([])
        if len(metadata_arr) > 0:
            if fit:
                metadata_scaled = self.scaler.fit_transform(metadata_arr)
            else:
                metadata_scaled = self.scaler.transform(metadata_arr)
            return hstack([X_word, X_char, csr_matrix(metadata_scaled)])
        
        return hstack([X_word, X_char])
    
    def fit(self, df, y):
        # Training
        X = self.extract_features(df, fit=True)
        self.model.fit(X, y)
        return self
    
    def predict(self, df):
        # Predictions using tuned threshold
        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, df):
        # Prediction probabilities
        X = self.extract_features(df, fit=False)
        return self.model.predict_proba(X)[:, 1]
    
    def tune_threshold(self, val_df, val_y):
        # Find optimal threshold for F1
        proba = self.predict_proba(val_df)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.30, 0.70, 0.01):
            preds = (proba >= threshold).astype(int)
            f1 = f1_score(val_y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
        return best_threshold
    
    def get_feature_importance(self, n=15):
        # Top predictive features
        coefs = self.model.coef_[0]
        n_word = len(self.tfidf_word.vocabulary_)
        text_coefs = coefs[:n_word]
        feature_names = self.tfidf_word.get_feature_names_out()
        
        top_pos_idx = np.argsort(text_coefs)[-n:][::-1]
        top_neg_idx = np.argsort(text_coefs)[:n]
        
        results = {
            'helpful': [(feature_names[i], text_coefs[i]) for i in top_pos_idx],
            'not_helpful': [(feature_names[i], text_coefs[i]) for i in top_neg_idx]
        }
        
        # Add metadata coefficients
        if len(self.metadata_features) > 0:
            n_char = len(self.tfidf_char.vocabulary_)
            metadata_coefs = coefs[n_word + n_char:]
            results['metadata'] = list(zip(self.metadata_features, metadata_coefs))
        
        return results


# Loading the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} reviews")
    return df


# Split data by pre-assigned splits
def get_splits(df):
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df


def evaluate_model(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }


# Calculate majority class baseline
def calculate_baseline(y_train, y_test):
    majority = np.bincount(y_train).argmax()
    baseline_pred = np.full_like(y_test, majority)
    
    f1_maj = f1_score(y_test, baseline_pred, zero_division=0)
    f1_min = f1_score(y_test, 1 - baseline_pred, zero_division=0)
    
    return max(f1_maj, f1_min)


def print_results(train_metrics, test_metrics, baseline_f1):
    print("\nF1 Score:")
    print(f"Training F1:   {train_metrics['f1']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Baseline F1:   {baseline_f1:.4f}")
    
    diff = (test_metrics['f1'] - baseline_f1) * 100
    print(f"Improvement:   {diff:+.2f}%")
    
    print("\nOther Metrics:")
    print(f"Accuracy:  Train={train_metrics['accuracy']:.4f}  Test={test_metrics['accuracy']:.4f}")
    print(f"Precision: Train={train_metrics['precision']:.4f}  Test={test_metrics['precision']:.4f}")
    print(f"Recall:    Train={train_metrics['recall']:.4f}  Test={test_metrics['recall']:.4f}")


def print_feature_importance(model, n=10):
    importance = model.get_feature_importance(n=n)
    
    print("Most Influential Words:")
    
    print("Predicted Helpful:")
    for word, coef in importance['helpful']:
        print(f"  {word:20s} {coef:+.4f}")
    
    print("Predicted NOT Helpful:")
    for word, coef in importance['not_helpful']:
        print(f"  {word:20s} {coef:+.4f}")
    
    if 'metadata' in importance:
        print("Metadata Features:")
        for feature, coef in importance['metadata']:
            print(f"  {feature:25s} {coef:+.4f}")


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Helpful', 'Helpful'],
                yticklabels=['Not Helpful', 'Helpful'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_f1(train_metrics, test_metrics, baseline_f1):
    # Plot F1 score with baseline
    x = np.array([0])
    width = 0.35
    
    plt.figure(figsize=(6, 6))
    plt.bar(x - width/2, [train_metrics['f1']], width, label='Train', alpha=0.8)
    plt.bar(x + width/2, [test_metrics['f1']], width, label='Test', alpha=0.8)
    plt.axhline(y=baseline_f1, color='r', linestyle='--', label=f'Baseline={baseline_f1:.3f}', alpha=0.7)
    
    plt.ylabel('Score')
    plt.title('F1 Score')
    plt.xticks([0], ['F1'])
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_other_metrics(train_metrics, test_metrics):
    # Plot other metrics without baseline
    metrics = ['Accuracy', 'Precision', 'Recall']
    train = [train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall']]
    test = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, train, width, label='Train', alpha=0.8)
    plt.bar(x + width/2, test, width, label='Test', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Other Performance Metrics')
    plt.xticks(x, metrics)
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_error_stats(test_df, y_test, y_pred):
    results = test_df.copy()
    results['pred'] = y_pred
    results['len'] = results['review'].str.len()
    
    tp = results[(y_test == 1) & (y_pred == 1)]
    tn = results[(y_test == 0) & (y_pred == 0)]
    fp = results[(y_test == 0) & (y_pred == 1)]
    fn = results[(y_test == 1) & (y_pred == 0)]
    
    print("Error Analysis Stats:")
    print(f"TP: {tp['len'].mean():.0f} avg chars, {(tp['len']<100).mean()*100:.1f}% under 100")
    print(f"TN: {tn['len'].mean():.0f} avg chars, {(tn['len']<100).mean()*100:.1f}% under 100")
    print(f"FP: {fp['len'].mean():.0f} avg chars, {(fp['len']<100).mean()*100:.1f}% under 100")
    print(f"FN: {fn['len'].mean():.0f} avg chars, {(fn['len']<100).mean()*100:.1f}% under 100")

def main():
    print("Steam Review Helpfulness Prediction\nGroup 150")
    
    # Load data
    df = load_data('Datasets/steam_reviews_100k_balanced.csv')
    
    # Get splits
    train_df, val_df, test_df = get_splits(df)
    
    # Extract labels
    y_train = train_df['label_helpful'].values
    y_val = val_df['label_helpful'].values
    y_test = test_df['label_helpful'].values
    
    # Train model
    print("\nTraining model:")
    model = Model(C=0.1, solver='lbfgs', class_weight='balanced')
    model.fit(train_df, y_train)
    
    # Tune threshold on validation set
    print("\nTuning threshold:")
    model.tune_threshold(val_df, y_val)
    
    # Make predictions
    y_train_pred = model.predict(train_df)
    y_test_pred = model.predict(test_df)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    baseline_f1 = calculate_baseline(y_train, y_test)
    
    # Print results
    print_results(train_metrics, test_metrics, baseline_f1)
    print_error_stats(test_df, y_test, y_test_pred)
    
    # Features
    print_feature_importance(model, n=10)
    
    # Graphs
    print("Graphs:")
    plot_f1(train_metrics, test_metrics, baseline_f1)
    plot_other_metrics(train_metrics, test_metrics)
    plot_confusion_matrix(y_test, y_test_pred)


if __name__ == '__main__':
    main()