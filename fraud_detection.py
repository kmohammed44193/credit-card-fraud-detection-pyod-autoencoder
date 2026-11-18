"""Unsupervised credit-card fraud detection with PyOD AutoEncoder.

Dataset: Kaggle 'fraud-detection' (creditcard.csv), with Class 0/1.
Run: python fraud_detection.py --data creditcard.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def run(data_path: Path, output_dir: Path, epochs: int, contamination: float) -> None:
    if not 0 < contamination < 0.5:
        raise ValueError('contamination must be between 0 and 0.5')
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    if 'Class' not in df.columns:
        raise ValueError('Expected target column Class (0=normal, 1=fraud).')
    y = df['Class'].astype(int)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() < 2:
        raise ValueError('Class must contain both 0 and 1.')
    X = df.drop(columns=['Class']).select_dtypes(include='number')
    if X.empty or X.isna().any().any():
        raise ValueError('Numeric features must exist and contain no missing values.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    # Fit scaler only on legitimate training transactions to prevent leakage.
    X_normal = X_train.loc[y_train == 0]
    scaler = StandardScaler().fit(X_normal)
    X_train_normal = scaler.transform(X_normal).astype('float32')
    X_test_scaled = scaler.transform(X_test).astype('float32')

    # Unsupervised training: model learns normal behavior, not fraud labels.
    model = AutoEncoder(
        hidden_neuron_list=[32, 16, 8, 16, 32],
       epoch_num=epochs, batch_size=256, contamination=contamination,
        verbose=1, random_state=SEED
    )
    model.fit(X_train_normal)
    train_scores = model.decision_scores_
    threshold = float(np.quantile(train_scores, 1 - contamination))
    scores = model.decision_function(X_test_scaled)
    predictions = (scores > threshold).astype(int)

    metrics = {
        'train_rows': int(len(X_train)), 'normal_train_rows': int(len(X_normal)),
        'test_rows': int(len(X_test)), 'fraud_test_rows': int(y_test.sum()),
        'threshold': threshold, 'contamination': contamination,
        'roc_auc': float(roc_auc_score(y_test, scores)),
        'average_precision': float(average_precision_score(y_test, scores)),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'classification_report': classification_report(y_test, predictions,
                                                          output_dict=True, zero_division=0),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({'actual': y_test.to_numpy(), 'anomaly_score': scores,
                  'predicted': predictions}).to_csv(output_dir / 'predictions.csv', index=False)
    print('Fraud Detection Results')
    print(f'Train rows: {len(X_train):,}; Test rows: {len(X_test):,}')
    print(f'Threshold: {threshold:.6f}')
    print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
    print(f'Average precision: {metrics["average_precision"]:.4f}')
    print(classification_report(y_test, predictions, zero_division=0))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap='Blues',
                                            display_labels=['Normal', 'Fraud'])
    plt.title('AutoEncoder Fraud Detection – Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, scores)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title('Precision–Recall Curve')
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall.png', dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for label, name in [(0, 'Normal'), (1, 'Fraud')]:
        subset = scores[y_test.to_numpy() == label]
        plt.hist(subset, bins=60, alpha=0.55, label=name, density=True)
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.yscale('log')
    plt.xlabel('Anomaly score (higher = more unusual)')
    plt.ylabel('Log density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_scores.png', dpi=180)
    plt.close()
    print(f'Output saved to {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--contamination', type=float, default=0.01)
    args = parser.parse_args()
    run(args.data, args.output, args.epochs, args.contamination)
