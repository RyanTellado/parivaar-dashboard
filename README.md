# Parivaar Probabilistic Clinical Dashboard

A Bayesian clinical decision-support prototype for rural healthcare workers — **using probability models to suggest likely diagnoses and recommend treatments based on patient outcomes data.**

Built on top of a medical informatics platform I developed for Parivaar, a humanitarian nonprofit serving 500,000+ patients across 17+ districts in rural India and South Sudan. The platform digitized patient records that were previously tracked with paper, pencil, and WhatsApp. This dashboard is the next layer: a probabilistic prediction and treatment recommendation engine built on top of that data.

## Modules

**Module 1 — Differential Diagnosis**
Uses a Naive Bayes classifier to compute a posterior probability distribution over 6 diagnoses given a patient's observed symptoms, age, and malnutrition status.

**Module 2 — Treatment Success Estimator**
Uses a Beta-Bernoulli conjugate model to estimate the probability that a given treatment succeeds for a specific patient, filtered by age group, district, and nutritional status. Displays the full posterior PDF with a 95% credible interval.

## Probability Models Used
Bayes' Theorem, Naive Bayes, MLE, Beta-Bernoulli Conjugate Model, Gaussian Mixture, Poisson, Multinomial, Bernoulli, Credible Intervals, Log-space Inference

## Live Demo

🌐 [medical-probability-dashboard.streamlit.app](https://medical-probability-dashboard.streamlit.app)

## Related
- [Parivaar App Platform (GitHub)](https://github.com/RyanTellado/MedicalApp-Showcase)

## Stack
Python · NumPy · SciPy · Streamlit · Pandas · Matplotlib
