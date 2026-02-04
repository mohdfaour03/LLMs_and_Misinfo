---
name: TVS-Validation-Skill
description: A specialized skill for performing rigorous validation, stress testing, and empirical optimization of the Trust Verification Score (TVS) for medical misinformation detection.
---

# TVS Validation Skill

This skill provides a structured framework for evaluating and optimizing the Trust Verification Score (TVS).

## Capabilities
- **Stress Testing**: Generating and evaluating synthetic "worst-case" misinformation scenarios.
- **Sweep Analysis**: Empirical testing of weight parameters (α, β, γ, δ) to maximize contrast between safe and unsafe responses.
- **Justification Logic**: Documenting the reasoning for each metric component (Hedging, Risk, Moderation, Grounding).

## Usage Protocol
1. **Identify Target Metric**: Current TVS formula in `combined_metric.py`.
2. **Execute Stress Test**: Run `tvs_stress_test.py` against edge cases.
3. **Run Sweep Analysis**: Iterate through weight combinations to find the highest empirical separation.
4. **Document Results**: Update `walkthrough.md` and `README.md` with findings.
5. **Worst-Case Evaluation**: Specifically measure responses that entail "stopping medication" (Moderation failure) or "hallucinating efficacy" (Grounding failure).

## Parameters to Optimize
- `ALPHA_H` (Hedging)
- `BETA_R` (Risk Coverage)
- `GAMMA_M` (Moderation)
- `DELTA_N` (Grounding/Novelty)
