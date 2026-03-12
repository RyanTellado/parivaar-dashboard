"""
models.py
---------
Probability models for the Parivaar clinical dashboard.

Module 1: NaiveBayesClassifier
  Implements P(diagnosis | symptoms) ∝ P(diagnosis) × ∏ P(symptom_i | diagnosis)
  using MLE with Laplace smoothing. No sklearn — all math is explicit.

Module 2: BetaBernoulliModel
  Models treatment success as Bernoulli(θ) with a Beta conjugate prior.
  Posterior update: Beta(α, β) + (s successes, f failures) → Beta(α+s, β+f)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from typing import Optional

# ---------------------------------------------------------------------------
# Shared constants (must match generate_data.py)
# ---------------------------------------------------------------------------
DIAGNOSES = ["Malaria", "Typhoid", "Tuberculosis", "Anemia", "ARI", "Dengue"]

SYMPTOMS = [
    "fever", "chills", "fatigue", "cough", "night_sweats",
    "headache", "nausea", "joint_pain", "pale_complexion",
    "shortness_of_breath", "abdominal_pain", "weight_loss"
]

TREATMENTS = {
    "Malaria":       ["Artemisinin Combination Therapy (ACT)", "Chloroquine"],
    "Typhoid":       ["Azithromycin", "Ciprofloxacin"],
    "Tuberculosis":  ["Standard DOTS", "Intensive Regimen"],
    "Anemia":        ["Iron + Folic Acid", "Dietary Counseling Only"],
    "ARI":           ["Amoxicillin", "Supportive Care Only"],
    "Dengue":        ["Supportive Care + Hydration", "Antipyretics + Monitoring"],
}

# Symptoms that get a +0.15 probability boost for malnourished patients
MALNUT_BOOST_SYMPTOMS = {"fatigue", "pale_complexion", "weight_loss"}

# ---------------------------------------------------------------------------
# Age-group prior adjustments for Module 1 (Naive Bayes)
# ---------------------------------------------------------------------------
# These are additive shifts applied to log P(diagnosis) before inference.
# A value of +k multiplies the effective prior by e^k ≈ 1+k for small k.
#
# Medical rationale:
#   child   : ARI and Malaria disproportionately affect children;
#             TB and severe Typhoid are rare under 18.
#   adult   : population-level priors from data (no adjustment).
#   elderly : TB reactivation and Typhoid are more common; acute febrile
#             illnesses like ARI are relatively less dominant.
#
# Diagnoses order: Malaria, Typhoid, Tuberculosis, Anemia, ARI, Dengue
AGE_LOG_PRIOR_ADJUSTMENTS: dict[str, np.ndarray] = {
    "child":   np.array([+0.30, -0.25, -0.60, +0.20, +0.55, +0.10]),
    "adult":   np.zeros(6),
    "elderly": np.array([-0.15, +0.20, +0.45, +0.15, -0.25, -0.10]),
}

# Color palette: deep teal + warm amber
TEAL   = "#0d6e6e"
AMBER  = "#e8a838"
TEAL_LIGHT  = "#4da8a8"
AMBER_LIGHT = "#f5cc80"
BG_COLOR    = "#f7f9f9"


# ===========================================================================
# Module 1: Naive Bayes Classifier
# ===========================================================================

class NaiveBayesClassifier:
    """
    Naive Bayes for differential diagnosis.

    Generative model assumption (CS109 Lecture 3 — Independence):
      P(s_1, ..., s_12 | diagnosis) = ∏_i P(s_i | diagnosis)
      i.e., symptoms are conditionally independent given the diagnosis.

    Bayes' theorem (CS109 Lecture 2):
      P(diagnosis | symptoms) ∝ P(diagnosis) × ∏_i P(s_i | diagnosis)

    Parameters are estimated from data using MLE (CS109 Lecture 19).
    """

    def __init__(self):
        self.log_priors: Optional[np.ndarray] = None          # shape (6,)
        # log_likelihoods_by_malnut[m][symptom, diagnosis, value]
        # m ∈ {0, 1} — estimated separately for non-malnourished / malnourished patients
        self.log_likelihoods_by_malnut: dict = {}
        self.diagnoses = DIAGNOSES
        self.symptoms  = SYMPTOMS
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "NaiveBayesClassifier":
        """
        Estimate model parameters from training data using MLE.

        Priors P(diagnosis): frequency counts (MLE)
          P̂(d) = count(d) / N

        Likelihoods P(symptom=1 | diagnosis): Laplace-smoothed MLE
          P̂(s=1 | d) = (count(s=1 ∧ d) + 1) / (count(d) + 2)
          P̂(s=0 | d) = 1 − P̂(s=1 | d)

        Laplace smoothing (add-1) prevents zero probabilities when a symptom
        is never observed for a given diagnosis in the training data.
        """
        n_diag  = len(self.diagnoses)
        n_sym   = len(self.symptoms)
        n_total = len(df)

        # --- Priors: MLE frequency counts (marginal over malnutrition status) ---
        prior_counts = np.array([
            (df["diagnosis"] == d).sum() for d in self.diagnoses
        ], dtype=float)
        self.log_priors = np.log(prior_counts / n_total)

        # --- Likelihoods: Laplace-smoothed MLE, fit separately for each
        #     malnutrition stratum so inference can condition on it.
        #
        #     P̂(s=1 | d, malnut=m) = (count(s=1, d, malnut=m) + 1)
        #                             / (count(d, malnut=m) + 2)
        #
        #     This is the principled way to extend Naive Bayes with an
        #     additional discrete covariate: condition the likelihoods on it.
        # -------------------------------------------------------------------
        for malnut_val in (0, 1):
            sub = df[df["malnourished"] == malnut_val]
            log_liks = np.zeros((n_sym, n_diag, 2))

            for d_idx, diagnosis in enumerate(self.diagnoses):
                diag_mask  = sub["diagnosis"] == diagnosis
                diag_count = diag_mask.sum()

                for s_idx, symptom in enumerate(self.symptoms):
                    positive_count = (sub.loc[diag_mask, symptom] == 1).sum()
                    p = (positive_count + 1) / (diag_count + 2)
                    log_liks[s_idx, d_idx, 1] = np.log(p)
                    log_liks[s_idx, d_idx, 0] = np.log(1.0 - p)

            self.log_likelihoods_by_malnut[malnut_val] = log_liks

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        symptom_values: dict,
        malnourished: bool = False,
        age_group: Optional[str] = None,
    ) -> dict:
        """
        Compute posterior P(diagnosis | symptoms, malnourished, age_group).

        Args:
            symptom_values: {symptom_name: 0 or 1} for all 12 symptoms
            malnourished:   use malnourished-stratum likelihoods when True
            age_group:      "child" | "adult" | "elderly" — applies log-prior
                            adjustment from AGE_LOG_PRIOR_ADJUSTMENTS

        Math (log-space):
          log P(d | s, m, a) ∝ [log P(d) + age_adj(d, a)]
                               + Σ_i log P(s_i | d, m)

          1. Start with data-estimated log priors.
          2. Add the age-group adjustment (shifts the prior toward age-typical
             disease distributions using domain knowledge).
          3. Accumulate log-likelihoods from the malnutrition stratum.
          4. Subtract max for numerical stability, exponentiate, normalize.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        # Step 1 + 2: age-adjusted log priors
        log_posteriors = self.log_priors.copy()  # shape (6,)
        if age_group is not None and age_group in AGE_LOG_PRIOR_ADJUSTMENTS:
            log_posteriors = log_posteriors + AGE_LOG_PRIOR_ADJUSTMENTS[age_group]

        # Step 3: accumulate malnutrition-stratum log-likelihoods
        log_liks = self.log_likelihoods_by_malnut[int(malnourished)]
        for s_idx, symptom in enumerate(self.symptoms):
            value = int(symptom_values.get(symptom, 0))
            log_posteriors += log_liks[s_idx, :, value]

        # Step 4: normalize
        log_posteriors -= log_posteriors.max()
        posteriors = np.exp(log_posteriors)
        posteriors /= posteriors.sum()

        return {diag: float(prob) for diag, prob in zip(self.diagnoses, posteriors)}

    def get_smoothed_likelihoods(self, malnourished: bool = False) -> pd.DataFrame:
        """Return P(symptom=1 | diagnosis, malnourished) for display."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before get_smoothed_likelihoods().")
        log_liks = self.log_likelihoods_by_malnut[int(malnourished)]
        probs = np.exp(log_liks[:, :, 1])
        return pd.DataFrame(probs, index=self.symptoms, columns=self.diagnoses)


# ===========================================================================
# Module 2: Beta-Bernoulli Treatment Success Estimator
# ===========================================================================

class BetaBernoulliModel:
    """
    Bayesian estimator for treatment success probability.

    Model:
      θ ~ Beta(α₀, β₀)          — conjugate prior
      outcome_i | θ ~ Bernoulli(θ)

    Posterior after observing s successes and f failures (CS109 Lecture 14):
      θ | data ~ Beta(α₀ + s, β₀ + f)

    With prior Beta(2, 2):
      - Symmetric around 0.5
      - Weak: expresses mild uncertainty before seeing data
      - As data accumulates the posterior concentrates around the true θ

    Conjugacy means the posterior is also Beta — closed-form, no MCMC needed.
    """

    # Weak symmetric prior: Beta(2, 2), mean = 0.5, variance = 1/20
    PRIOR_ALPHA = 2.0
    PRIOR_BETA  = 2.0

    def get_cohort(
        self,
        df: pd.DataFrame,
        diagnosis: str,
        treatment: str,
        age_group: Optional[str] = None,
        malnourished: Optional[bool] = None,
        district: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Filter dataset to the relevant cohort: same diagnosis + same treatment,
        optionally restricted by age group, malnutrition status, and/or district.

        Age groups: "child" (< 18), "adult" (18–60), "elderly" (> 60)
        malnourished: True → only malnourished patients
        district: integer 1–17 → filter to that district only
        """
        mask = (df["diagnosis"] == diagnosis) & (df["treatment"] == treatment)
        cohort = df[mask].copy()

        if age_group == "child":
            cohort = cohort[cohort["age"] < 18]
        elif age_group == "adult":
            cohort = cohort[(cohort["age"] >= 18) & (cohort["age"] <= 60)]
        elif age_group == "elderly":
            cohort = cohort[cohort["age"] > 60]

        if malnourished is not None:
            cohort = cohort[cohort["malnourished"] == int(malnourished)]

        if district is not None:
            cohort = cohort[cohort["district"] == int(district)]

        return cohort

    def get_posterior_params(
        self,
        df: pd.DataFrame,
        diagnosis: str,
        treatment: str,
        age_group: Optional[str] = None,
        malnourished: Optional[bool] = None,
        district: Optional[int] = None,
    ) -> tuple[float, float, int]:
        """
        Compute posterior Beta parameters from observed outcomes.

        Returns:
            (alpha_post, beta_post, n_patients)

        Posterior update rule:
          alpha_post = α₀ + successes
          beta_post  = β₀ + failures
        """
        cohort    = self.get_cohort(df, diagnosis, treatment, age_group, malnourished, district)
        n         = len(cohort)
        successes = cohort["outcome"].sum()
        failures  = n - successes

        alpha_post = self.PRIOR_ALPHA + successes
        beta_post  = self.PRIOR_BETA  + failures

        return float(alpha_post), float(beta_post), int(n)

    def credible_interval(
        self,
        alpha: float,
        beta: float,
        level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute the highest-density credible interval for Beta(alpha, beta).

        For a unimodal Beta distribution, the equal-tailed interval from the
        quantile function is a good approximation and exactly computable.

        Returns: (lower, upper) bounds of the interval.
        """
        tail = (1.0 - level) / 2.0
        lower = stats.beta.ppf(tail,       alpha, beta)
        upper = stats.beta.ppf(1.0 - tail, alpha, beta)
        return float(lower), float(upper)

    def posterior_mean(self, alpha: float, beta: float) -> float:
        """
        Mean of Beta(α, β) = α / (α + β)
        This is the point estimate for treatment success probability.
        """
        return alpha / (alpha + beta)

    def posterior_variance(self, alpha: float, beta: float) -> float:
        """
        Variance of Beta(α, β) = αβ / [(α+β)²(α+β+1)]
        Shown in the UI to demonstrate how certainty grows with sample size.
        """
        total = alpha + beta
        return (alpha * beta) / (total ** 2 * (total + 1))

    def plot_posterior(
        self,
        alpha: float,
        beta: float,
        n_patients: int,
        diagnosis: str,
        treatment: str,
        age_group: Optional[str] = None,
        malnourished: Optional[bool] = None,
        comparison_params: Optional[tuple[float, float, int]] = None,
        comparison_label: str = "All patients",
    ) -> plt.Figure:
        """
        Plot the Beta posterior PDF with:
          - Shaded 95% credible interval
          - Mean marked with vertical dashed line
          - Optional comparison curve (age-group vs. full population)

        Args:
            alpha, beta:         Posterior parameters for the current view
            n_patients:          Cohort size (shown in title/legend)
            diagnosis, treatment: For display labels
            age_group:           Label for current filter (None = all patients)
            comparison_params:   (alpha, beta, n) for the unfiltered posterior
                                 — shown as a faded curve for contrast
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Build x-axis: [0, 1] with extra density near the mean
        x = np.linspace(0.001, 0.999, 500)

        # Build a human-readable cohort label for the legend
        cohort_parts = []
        if age_group is not None:
            cohort_parts.append(age_group)
        if malnourished is True:
            cohort_parts.append("malnourished")
        elif malnourished is False:
            cohort_parts.append("well-nourished")
        cohort_label = ", ".join(cohort_parts) if cohort_parts else "all patients"

        # --- Main posterior curve ---
        y = stats.beta.pdf(x, alpha, beta)
        ax.plot(x, y, color=TEAL, linewidth=2.5, zorder=3,
                label=f"Posterior ({cohort_label}, n={n_patients})")

        # Shade 95% credible interval
        lo, hi = self.credible_interval(alpha, beta)
        ci_mask = (x >= lo) & (x <= hi)
        ax.fill_between(x[ci_mask], y[ci_mask], alpha=0.25, color=TEAL, zorder=2,
                        label=f"95% CI [{lo:.2f}, {hi:.2f}]")

        # Mark the posterior mean
        mean_val = self.posterior_mean(alpha, beta)
        ax.axvline(mean_val, color=AMBER, linewidth=2, linestyle="--", zorder=4,
                   label=f"Mean = {mean_val:.3f}")

        # --- Optional comparison curve (filtered vs. full population) ---
        if comparison_params is not None:
            a2, b2, n2 = comparison_params
            y2 = stats.beta.pdf(x, a2, b2)
            mean2 = self.posterior_mean(a2, b2)
            ax.plot(x, y2, color=TEAL_LIGHT, linewidth=1.8, linestyle=":", zorder=2,
                    alpha=0.7, label=f"{comparison_label} (n={n2}), mean={mean2:.3f}")

        # --- Styling ---
        ax.set_xlabel("Treatment Success Probability (θ)", fontsize=11, color="#333")
        ax.set_ylabel("Posterior Density", fontsize=11, color="#333")
        ax.set_title(
            f"Beta Posterior — {treatment}\n"
            f"Beta({alpha:.0f}, {beta:.0f})  ·  Prior: Beta(2, 2)",
            fontsize=11, color="#222", pad=10
        )
        ax.tick_params(colors="#555", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccc")

        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5, framealpha=0.7, loc="upper left")

        # Annotate CI boundaries with dashed grey lines
        for boundary in [lo, hi]:
            ax.axvline(boundary, color="#aaa", linewidth=0.9,
                       linestyle="--", zorder=1, alpha=0.6)

        fig.tight_layout()
        return fig
