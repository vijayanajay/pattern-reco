# Pattern-Reco Constitution — Hard Rules for Code & Spec Generation

These are 20 non-negotiable rules for generating code, specs, and artifacts in this repository. They are inspired by two guiding mindsets: Kailash Nadh's emphasis on simplicity and explainability, and Geoffrey Hinton's insistence on statistical rigor and skepticism. Follow these rules exactly; deviations must be documented and approved.

1. Minimalism First: Prefer the smallest, simplest solution that answers the research question. Avoid additional features or abstractions unless they clearly enable the core experiment.

2. One Responsibility Per Artifact: Each code module, function, and spec section must have a single, well-documented responsibility. If a file does more than one thing, split it.

3. Test-First for Behavior: For any non-trivial computation (SMA calculation, signal detection, forward-return or drawdown computation, aggregation, t-test), write unit tests before changing behavior. Tests define acceptable numerical tolerance and edge-case behavior.

4. Reproducible Data Paths: All scripts must accept explicit input/output paths as parameters; no hard-coded network endpoints or implicit filesystem locations. Default paths may be provided but must be overridable.

5. Deterministic by Default: Analysis must be deterministic given the same input data and parameters. Any stochastic component must either be removed or have a fixed seed provided and recorded in outputs.

6. Explicit Data Hygiene: Always publish the data-cleaning steps applied (e.g., forward-fill counts, NaN thresholds, rows dropped). Warnings for filled NaNs must be printed and recorded in a log artifact.

7. Fail Fast on Ambiguity: If a function or pipeline step cannot compute a value unambiguously (e.g., insufficient forward days for a holding period), it must flag and discard the case with a clear reason. Do not silently impute or extrapolate beyond documented rules.

8. No Hidden Defaults: Every default parameter that materially affects results (e.g., p-value threshold, holding-period days, SMA pairs, date slices) must be declared in a single configuration file and documented in the spec.

9. Statistical Rigor: All reported summary statistics must include sampling information (Num_Signals) and an appropriate measure of uncertainty (std, p-value). Mark results with low statistical power (Num_Signals < 5) as low-confidence.

10. Use Standard Tests Properly: When applying hypothesis tests (e.g., one-sample t-test), document assumptions and validate them when possible (normality checks, or robust alternatives). Report test statistic, degrees of freedom, and two-sided p-value.

11. Conservative Significance Claims: A p-value < 0.05 is a helpful threshold but not a guarantee. Report effect sizes (mean return) alongside p-values and require both positive effect and p < 0.05 before flagging a parameter as "significant" in summaries.

12. Audit-Friendly Outputs: CSVs and reports must include provenance metadata (date run, git branch/commit, parameter set, data range). Save this metadata alongside the primary artifacts.

13. Human-Readable Reports: Executive summaries must state the bottom-line conclusion in plain language, then provide the quantitative evidence and caveats. Visuals (heatmaps) should use color scales centered at zero and annotated with sample counts when ambiguous.

14. No Black-Box Magic: Do not rely on opaque, proprietary, or un-auditable code paths for core computations. Prefer transparent, easily inspectable implementations for SMA, returns, and drawdown calculations.

15. Edge-Case Transparency: When including a special case study (e.g., YESBANK.NS), document why it's special, what data limitations exist, and how conclusions are tempered by those limitations.

16. Traceable Repro Steps: A single command or script must reproduce the full analysis from raw data to final `sma_crossover_analysis_results.csv`, `heatmap_avg_mean_return.png`, and `analysis_summary.md`. Provide a minimal README for that command.

17. Validate Outputs with Quick Sanity Tests: After running the analysis, execute a small suite of smoke checks (files created, non-empty CSV rows, no NaNs in key columns, p-values within [0,1]) and fail the run if any check fails.

18. Version & Change Log: Every spec change that affects behavior must include a brief change note (what changed and why) and a version increment in the spec header.

19. Preserve Simplicity in Interfaces: Public functions should have a small number of clear parameters (inputs, outputs, options). Complex behavior must be composed from simple building blocks, not monolithic functions with many toggles.

20. Ethical and Interpretability Reminder: Avoid overclaiming. All conclusions must be presented with uncertainty bounds and an explicit statement that this is an exploratory analysis, not investment advice.

**Version**: 1.0 | **Ratified**: 2025-10-18 | **Last Amended**: 2025-10-18

