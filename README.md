# MoodSignal

NLP-based classification of bipolar disorder, depression, and anxiety from text.

Built by Anshika Zakkireddy and Varish Matlapudi — AI/ML Workshop Project 2026.

## What it does

**Module 1 — Condition Classifier**
Takes free text and classifies it as bipolar, depression, or anxiety
using a fine-tuned MentalBERT model with SHAP explainability.

**Module 2 — Episode Tracker**
Takes daily journal entries over multiple days, classifies each day
as manic, hypomanic, depressive, or stable, and detects mood trajectory
patterns over time.

> This is a flag-and-refer tool. It never diagnoses.
> All outputs recommend professional consultation.

## Results

| Model | Accuracy | Bipolar F1 | Depression F1 | Anxiety F1 |
|---|---|---|---|---|
| Baseline TF-IDF + LR | 85.5% | 0.85 | 0.83 | 0.88 |
| BiLSTM | TBD | TBD | TBD | TBD |
| MentalBERT | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |

## Dataset

Reddit mental health posts — 3,367 posts across bipolar, depression, anxiety.
Source: Kaggle Reddit Mental Health Dataset

## Project structure

notebooks/ — all 14 code files in order
data/ — cleaned datasets (not uploaded)
models/ — trained models (not uploaded, too large)
outputs/ — charts and evaluation results

## Setup

pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets shap imbalanced-learn nltk

Run notebooks in order from 01 to 14.

## Team

A — files 06, 08, 10, 12, 14
V — files 07, 09, 11, 13

## Ethics statement

Dataset uses self-reported Reddit labels.
SMOTE applied to training data only.
Not validated on clinical populations.
See paper for full limitations.