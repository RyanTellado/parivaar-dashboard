"""
generate_data.py
----------------
Generates 10,000 synthetic patient records for the Parivaar clinical dashboard.

Probability concepts used:
  - Gaussian mixture model for age (CS109 Lecture 9)
  - Bernoulli sampling for binary features (CS109 Lecture 5)
  - Multinomial sampling for diagnosis and district (CS109 Lecture 13)
  - Poisson distribution for prior visit counts (CS109 Lecture 7)
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N_PATIENTS = 10_000

# ---------------------------------------------------------------------------
# Diagnosis prevalence (used as Multinomial probabilities)
# ---------------------------------------------------------------------------
DIAGNOSES = ["Malaria", "Typhoid", "Tuberculosis", "Anemia", "ARI", "Dengue"]
PREVALENCE = [0.28, 0.18, 0.12, 0.22, 0.14, 0.06]

# ---------------------------------------------------------------------------
# Conditional symptom probabilities P(symptom=1 | diagnosis)
# Rows = symptoms, Columns = [Malaria, Typhoid, TB, Anemia, ARI, Dengue]
# ---------------------------------------------------------------------------
SYMPTOMS = [
    "fever", "chills", "fatigue", "cough", "night_sweats",
    "headache", "nausea", "joint_pain", "pale_complexion",
    "shortness_of_breath", "abdominal_pain", "weight_loss"
]

# Shape: (12 symptoms, 6 diagnoses)
COND_PROBS = np.array([
    # Mal    Typ    TB    Ane    ARI   Den
    [0.95,  0.92,  0.60,  0.20,  0.75,  0.97],  # fever
    [0.88,  0.30,  0.15,  0.10,  0.20,  0.45],  # chills
    [0.80,  0.78,  0.90,  0.92,  0.60,  0.75],  # fatigue
    [0.25,  0.20,  0.95,  0.15,  0.88,  0.20],  # cough
    [0.40,  0.35,  0.85,  0.20,  0.10,  0.25],  # night_sweats
    [0.85,  0.80,  0.30,  0.40,  0.55,  0.90],  # headache
    [0.70,  0.65,  0.30,  0.25,  0.35,  0.68],  # nausea
    [0.75,  0.40,  0.25,  0.30,  0.35,  0.92],  # joint_pain
    [0.30,  0.25,  0.40,  0.88,  0.15,  0.20],  # pale_complexion
    [0.20,  0.15,  0.70,  0.65,  0.72,  0.20],  # shortness_of_breath
    [0.45,  0.70,  0.20,  0.30,  0.25,  0.50],  # abdominal_pain
    [0.20,  0.30,  0.88,  0.45,  0.15,  0.15],  # weight_loss
])

# Indices of the three malnutrition-boosted symptoms
MALNUT_BOOST_IDX = {
    "fatigue": 2,
    "pale_complexion": 8,
    "weight_loss": 11,
}
MALNUT_BOOST = 0.15

# ---------------------------------------------------------------------------
# Treatment options and base success probabilities
# ---------------------------------------------------------------------------
TREATMENTS = {
    "Malaria": [
        ("Artemisinin Combination Therapy (ACT)", 0.87),
        ("Chloroquine", 0.61),
    ],
    "Typhoid": [
        ("Azithromycin", 0.83),
        ("Ciprofloxacin", 0.79),
    ],
    "Tuberculosis": [
        ("Standard DOTS", 0.76),
        ("Intensive Regimen", 0.71),
    ],
    "Anemia": [
        ("Iron + Folic Acid", 0.80),
        ("Dietary Counseling Only", 0.52),
    ],
    "ARI": [
        ("Amoxicillin", 0.82),
        ("Supportive Care Only", 0.58),
    ],
    "Dengue": [
        ("Supportive Care + Hydration", 0.88),
        ("Antipyretics + Monitoring", 0.79),
    ],
}


def sample_age(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample ages from a Gaussian mixture:
      - 40% child:  N(8, 10²)  clipped to [0, 17]
      - 60% adult:  N(34, 12²) clipped to [18, 90]
    """
    # Bernoulli to decide which component each patient draws from
    is_child = rng.random(n) < 0.40

    child_ages  = rng.normal(loc=8,  scale=10, size=n).clip(0, 17)
    adult_ages  = rng.normal(loc=34, scale=12, size=n).clip(18, 90)

    ages = np.where(is_child, child_ages, adult_ages).astype(int)
    return ages


def sample_district(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample from 17 districts using a Multinomial.
    Districts 1–5 are larger (higher weight).
    """
    # Unequal weights: districts 1-5 get ~60% of traffic
    weights = np.array([
        0.12, 0.12, 0.12, 0.12, 0.12,   # larger districts (1-5)
        0.05, 0.05, 0.05, 0.05, 0.05,   # medium districts (6-10)
        0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01  # smaller (11-17)
    ])
    weights /= weights.sum()  # normalize to sum to 1
    districts = rng.choice(np.arange(1, 18), size=n, p=weights)
    return districts


def compute_success_prob(
    base_prob: float,
    age: int,
    malnourished: bool,
    prior_visits: int
) -> float:
    """
    Adjust base treatment success probability for patient demographics.

    Adjustments (CS109: Bernoulli parameter estimation):
      - Malnourished:        −0.12
      - Age < 5:             −0.08
      - Age > 60:            −0.10
      - Prior visits > 5:    +0.05
    Clamped to [0.05, 0.99].
    """
    p = base_prob
    if malnourished:
        p -= 0.12
    if age < 5:
        p -= 0.08
    if age > 60:
        p -= 0.10
    if prior_visits > 5:
        p += 0.05
    return float(np.clip(p, 0.05, 0.99))


def generate_patients(n: int = N_PATIENTS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # --- Demographics ---
    ages          = sample_age(rng, n)
    sexes         = np.where(rng.random(n) < 0.52, "F", "M")
    districts     = sample_district(rng, n)
    malnourished  = rng.random(n) < 0.38   # Bernoulli(0.38)
    # Poisson(λ=3) for prior visits, minimum 1
    prior_visits  = np.maximum(rng.poisson(lam=3, size=n), 1)

    # --- Diagnoses: Multinomial draw (one category per patient) ---
    diagnosis_idx = rng.choice(len(DIAGNOSES), size=n, p=PREVALENCE)
    diagnoses     = [DIAGNOSES[i] for i in diagnosis_idx]

    # --- Symptoms: Bernoulli draw per symptom per patient ---
    # Start with the conditional probability matrix sliced to each patient's diagnosis
    # cond_probs_per_patient shape: (n, 12)
    cond_probs_per_patient = COND_PROBS[:, diagnosis_idx].T  # (n, 12)

    # Apply malnutrition boost to fatigue, pale_complexion, weight_loss
    for sym_name, sym_idx in MALNUT_BOOST_IDX.items():
        cond_probs_per_patient[malnourished, sym_idx] = np.minimum(
            cond_probs_per_patient[malnourished, sym_idx] + MALNUT_BOOST, 1.0
        )

    # Sample symptoms: compare uniform draw to threshold probability
    symptom_data = (rng.random((n, 12)) < cond_probs_per_patient).astype(int)

    # --- Treatments: uniform random from each diagnosis's list ---
    treatment_names = []
    base_probs      = []
    for diag in diagnoses:
        options = TREATMENTS[diag]
        chosen  = options[rng.integers(0, len(options))]
        treatment_names.append(chosen[0])
        base_probs.append(chosen[1])

    # --- Outcomes: Bernoulli(success_prob) ---
    outcomes = []
    for i in range(n):
        p = compute_success_prob(
            base_probs[i], ages[i], malnourished[i], prior_visits[i]
        )
        outcomes.append(int(rng.random() < p))

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "patient_id":   [f"P{i:05d}" for i in range(n)],
        "age":          ages,
        "sex":          sexes,
        "district":     districts,
        "malnourished": malnourished.astype(int),
        "prior_visits": prior_visits,
        "diagnosis":    diagnoses,
        "treatment":    treatment_names,
        "outcome":      outcomes,
    })

    # Add symptom columns
    symptom_df = pd.DataFrame(symptom_data, columns=SYMPTOMS)
    df = pd.concat([df, symptom_df], axis=1)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print(f"Generating {N_PATIENTS} patient records...")
    df = generate_patients(N_PATIENTS)
    df.to_csv("data/patients.csv", index=False)
    print(f"Saved to data/patients.csv  ({len(df)} rows, {len(df.columns)} columns)")

    # Quick sanity checks
    print("\nDiagnosis distribution (should match prevalence):")
    print(df["diagnosis"].value_counts(normalize=True).round(3))
    print("\nMalnourished rate:", df["malnourished"].mean().round(3), "(expected ~0.38)")
    print("Outcome rate:", df["outcome"].mean().round(3))
    print("Age range:", df["age"].min(), "–", df["age"].max())
