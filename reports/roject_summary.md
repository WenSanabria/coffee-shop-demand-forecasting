# Academic Project Summary

This portfolio repository is a refactored, privacy-safe version of a graduate data science programming project.

## Original project

The academic project used approximately **286,000 private POS transaction records** from a small Denver coffee shop to forecast daily product demand. The raw business dataset is intentionally not included in this public repository.

The original workflow cleaned transaction data, standardized inconsistent menu-item names, aggregated demand to the daily item level, engineered lag/calendar/holiday/weather/price features, and compared a lag-1 baseline with Decision Tree, Random Forest, and Gradient Boosting regressors.

On the original private dataset, **Gradient Boosting achieved the lowest RMSE (6.98) with MAE 4.00**. These figures are academic-project results and should not be interpreted as performance on the synthetic public sample.

## Public portfolio version

The public version preserves the technical workflow while replacing private business data with a synthetic dataset that uses the same general schema. It also refactors the code into reusable functions, removes notebook installation cells, adds a reproducible command-line entry point, and makes the forecasting demo explicitly one-day-ahead to avoid implying unavailable future lag values.
