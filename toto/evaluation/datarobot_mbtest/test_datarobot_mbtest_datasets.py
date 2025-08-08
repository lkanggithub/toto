import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Standard library imports
import gc

# Third-party imports
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Local imports
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from toto.evaluation.datarobot_mbtest.data import TestDataset

from toto.inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.model.toto import Toto

from dr_model_benchmark.common.analysis.entities import ModelScoreMetrics
from dr_model_benchmark.common.analysis.entities import ModelTimeProfiles
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.profile.entities import Seconds
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from dr_model_benchmark.common.profile.enums import TimeProfileType


DEFAULT_CONTEXT_LENGTH = 4096

# Define metrics configuration once at module level
METRIC_CONFIGS = {
    "MAE": (lambda: MAE(), "MAE[0.5]"),
    "MSE": (lambda: MSE(forecast_type=0.5), "MSE[0.5]"),
    "MSE_MEAN": (lambda: MSE(forecast_type="mean"), "MSE[mean]"),
    "MASE": (lambda: MASE(), "MASE[0.5]"),
    "MAPE": (lambda: MAPE(), "MAPE[0.5]"),
    "SMAPE": (lambda: SMAPE(), "sMAPE[0.5]"),
    "MSIS": (lambda: MSIS(), "MSIS"),
    "RMSE": (lambda: RMSE(forecast_type=0.5), "RMSE[0.5]"),
    "RMSE_MEAN": (lambda: RMSE(forecast_type="mean"), "RMSE[mean]"),
    "NRMSE": (lambda: NRMSE(forecast_type=0.5), "NRMSE[0.5]"),
    "NRMSE_MEAN": (lambda: NRMSE(forecast_type="mean"), "NRMSE[mean]"),
    "ND": (lambda: ND(), "ND[0.5]"),
    "WQTL": (
        lambda: MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        "mean_weighted_sum_quantile_loss",
    ),
}


def get_total_gpu_memory():
    """Get total GPU VRAM capacity in MB."""
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)


def calculate_optimal_batch_size(
    model,
    target_dim,
    prediction_length,
    context_length,
    use_kv_cache,
    num_samples,
    safety_factor=0.01,
):
    """
    Calculate the optimal batch size based on available GPU memory and model requirements.

    Args:
        model: Pre-loaded TOTO model
        target_dim: Target dimensionality (number of variates)
        prediction_length: Length of prediction horizon
        context_length: Context window length
        use_kv_cache: Whether KV cache is used
        num_samples: Number of samples to generate
        safety_factor: Safety factor to apply when calculating available memory (default=0.01)

    Returns:
        Suggested batch size for prediction
    """

    try:
        # Extract model size information
        model_width = model.model.embed_dim  # Feature dimension
        model_depth = model.model.num_layers  # Number of transformer layers

        # Calculate model's parameter memory footprint in MB
        model_param_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        # Base memory per sample in MB (parameters + activations + gradients)
        base_memory_per_sample = (model_width * model_depth * 4) / (1024 * 1024)

        # Memory for input/output tensors
        io_memory = (target_dim * (context_length + prediction_length) * 4) / (
            1024 * 1024
        )

        # KV cache memory (if used)
        kv_memory = 0
        if use_kv_cache:
            kv_memory = (model_depth * model_width * 2 * context_length * 4) / (
                1024 * 1024
            )

        # Total memory per sample
        mem_per_sample_mb = base_memory_per_sample + io_memory + kv_memory

        # Factor in target dimensions and samples directly
        # Each dimension and sample has a direct multiplicative effect on memory
        mem_per_batch_mb = (
            mem_per_sample_mb * target_dim * num_samples
        )  # Total memory for a batch with num_samples samples

        # Get total GPU VRAM capacity and subtract model parameter memory
        gpu_mem = get_total_gpu_memory()  # in MB
        cuda_reserved_mb = 1024  # Reserve 1GB for CUDA runtime and other overhead

        # Available memory = (Total VRAM - Model parameters - CUDA reserved) * safety factor
        available_memory = (
            gpu_mem - model_param_memory_mb - cuda_reserved_mb
        ) * safety_factor

        # Calculate max batch size based on available memory
        max_batch_size = max(
            1, int(available_memory / (mem_per_batch_mb / num_samples))
        )

        max_batch_size = min(16, max_batch_size)
        return max_batch_size
    except RuntimeError as e:
        print(f"Error calculating optimal batch size: {e}")
        return 1


class TOTOModelPredictorWrapper:
    """Wrapper for TOTOPredictor that handles OOM errors by adjusting batch size."""

    def __init__(
        self,
        model,
        prediction_length,
        context_length,
        mode,
        num_samples=128,
        use_kv_cache=True,
    ):
        """
        Initialize the predictor wrapper with specified parameters.

        Args:
            model: The loaded TOTO model instance to use for predictions
            prediction_length: The length of the prediction horizon.
            context_length: The length of the context window.
            mode: Mode of prediction (e.g., "forecast").
            num_samples: Total number of samples to generate.
            use_kv_cache: Whether to use key-value caching.
        """

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.mode = mode
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = (
            num_samples  # Start with full batch size and adjust if needed
        )
        self.model = model
        self._adjusted = False  # Tracks whether adjustment has been done

        self._initialize_predictor()

    def _initialize_predictor(self):
        """
        Initialize the TOTOPredictor with the current samples_per_batch.
        """
        self.predictor = TotoPredictor.create_for_eval(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            mode=self.mode,
            samples_per_batch=self.samples_per_batch,
        )

    def predict(self, gluonts_test_data: tuple):
        """
        Perform prediction while adjusting num_samples, samples_per_batch, and context_length if OOM errors occur.
        """
        predictions = None

        # Adjust only on the first call.
        if not self._adjusted:

            print(
                "Initializing predictor with samples_per_batch =",
                self.samples_per_batch,
            )
            while self.samples_per_batch >= 1:
                try:
                    print(
                        f"Attempting prediction with samples_per_batch = {self.samples_per_batch} and context_length = {self.context_length}"
                    )
                    # Attempt prediction (consume the generator to catch any OOM)
                    predictions = list(
                        self.predictor.predict(
                            gluonts_test_data,
                            use_kv_cache=self.use_kv_cache,
                            num_samples=self.num_samples,
                        )
                    )
                    self._adjusted = True
                    return predictions  # Prediction succeeded

                except RuntimeError as e:
                    print("Caught exception during prediction:", e)
                    if "CUDA out of memory" in str(e):
                        # First, try reducing the batch size if possible.
                        if self.samples_per_batch > 1:
                            print(
                                f"Out of memory with samples_per_batch = {self.samples_per_batch}. Reducing batch size."
                            )
                            self.samples_per_batch = self.samples_per_batch // 2
                            # Clean up GPU memory before trying with smaller batch size
                            torch.cuda.empty_cache()
                        else:
                            # Cannot reduce batch size further, so we fail
                            print(
                                f"OOM at minimal batch size. Cannot proceed with this context length and sample count."
                            )
                            raise e
                        # Reinitialize the predictor with the new settings.
                        self._initialize_predictor()
                    else:
                        raise e  # Re-raise unexpected exceptions

        # For subsequent calls, simply return the generator.
        return self.predictor.predict(
            gluonts_test_data,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )


# Helper functions to reduce repeated logic
def init_metrics(optimization_metric=None):
    """Initialize metrics based on the optimization metric or all metrics."""
    if optimization_metric:
        # Only initialize the specific metric needed
        metric_factory, metric_key = METRIC_CONFIGS[optimization_metric]
        # Create the metric by calling the lambda
        metric_obj = metric_factory()
        return [metric_obj], metric_key
    else:
        # Create all metrics from the config
        return [factory() for factory, _ in METRIC_CONFIGS.values()], None


def try_prediction_with_config(
    model,
    prediction_length,
    context_length,
    mode,
    num_samples,
    test_data,
    freq,
    use_kv_cache,
    metrics,
    min_context_length=None,
):
    """
    Attempt prediction with a specific configuration, handling OOM errors.

    Args:
        model: The loaded model instance to use
        prediction_length: Prediction horizon length
        context_length: Context window length
        mode: Prediction mode
        num_samples: Number of samples to generate (fixed for evaluation)
        test_data: data to evaluate on
        freq: frequency of the data
        use_kv_cache: Whether to use key-value caching
        metrics: Metrics to evaluate
        min_context_length: Minimum allowed context length

    Returns:
        Metrics result if successful, None if OOM occurs and can't be resolved
    """
    # Get patch size if min_context_length not provided
    if min_context_length is None:
        min_context_length = model.model.patch_embed.stride

    """
    # Ensure context_length is not smaller than the minimum
    context_length = max(context_length, min_context_length)
    """

    # Use the TOTOModelPredictorWrapper
    predictor_wrapper = TOTOModelPredictorWrapper(
        model=model,
        prediction_length=prediction_length,
        context_length=context_length,
        mode=mode,
        num_samples=num_samples,
        use_kv_cache=use_kv_cache,
    )

    try:
        # Attempt prediction and evaluation
        res = evaluate_model(
            predictor_wrapper,
            test_data=test_data,
            metrics=metrics,
            axis=None,
            batch_size=num_samples,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(freq),
        )
        return res
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def evaluate_dataset_with_model(model, dataset: TestDataset) -> pd.DataFrame:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")

    print(f"Evaluating dataset {dataset.name}")

    # Get min context length from model
    min_context_length = model.model.patch_embed.stride
    print(f"Model min context length (patch size): {min_context_length}")

    context_length = dataset.prediction_length
    print(f"Model context length: {context_length}")

    # Set up evaluation metrics - create all metrics from the config
    metrics, _ = init_metrics()

    # Calculate optimal batch size based on available GPU memory, not used for prediction
    suggested_batch_size = calculate_optimal_batch_size(
        model=model,
        target_dim=dataset.target_dim,
        prediction_length=dataset.prediction_length,
        context_length=context_length,
        use_kv_cache=USE_KV_CACHE,
        num_samples=NUM_SAMPLES,
    )
    eval_data = dataset.test_data

    # Try prediction with the optimal parameters - pass loaded model directly
    res = try_prediction_with_config(
        model=model,
        prediction_length=dataset.prediction_length,
        context_length=context_length,
        mode=Multivariate(batch_size=suggested_batch_size),
        num_samples=NUM_SAMPLES,
        test_data=eval_data,
        freq=dataset.time_series_frequency,
        use_kv_cache=USE_KV_CACHE,
        metrics=metrics,
        min_context_length=min_context_length,
    )

    # Process results - check if prediction was successful
    if res is None:
        print(f"Prediction failed for {dataset.name}")
        # Return a DataFrame with just metadata but NaN for metrics
        return pd.DataFrame(
            {
                "dataset": [dataset.name],
                "eval_metrics/MSE[mean]": [float("nan")],
                "eval_metrics/MSE[0.5]": [float("nan")],
                "eval_metrics/MAE[0.5]": [float("nan")],
                "eval_metrics/MASE[0.5]": [float("nan")],
                "eval_metrics/MAPE[0.5]": [float("nan")],
                "eval_metrics/sMAPE[0.5]": [float("nan")],
                "eval_metrics/MSIS": [float("nan")],
                "eval_metrics/RMSE[mean]": [float("nan")],
                "eval_metrics/NRMSE[mean]": [float("nan")],
                "eval_metrics/ND[0.5]": [float("nan")],
                "eval_metrics/mean_weighted_sum_quantile_loss": [float("nan")],
            }
        )

    # Create result dataframe
    result_df = pd.DataFrame(
        {
            "dataset": [dataset.name],
            "eval_metrics/MSE[mean]": [res["MSE[mean]"][0]],
            "eval_metrics/MSE[0.5]": [res["MSE[0.5]"][0]],
            "eval_metrics/MAE[0.5]": [res["MAE[0.5]"][0]],
            "eval_metrics/MASE[0.5]": [res["MASE[0.5]"][0]],
            "eval_metrics/MAPE[0.5]": [res["MAPE[0.5]"][0]],
            "eval_metrics/sMAPE[0.5]": [res["sMAPE[0.5]"][0]],
            "eval_metrics/MSIS": [res["MSIS"][0]],
            "eval_metrics/RMSE[mean]": [res["RMSE[mean]"][0]],
            "eval_metrics/NRMSE[mean]": [res["NRMSE[mean]"][0]],
            "eval_metrics/ND[0.5]": [res["ND[0.5]"][0]],
            "eval_metrics/mean_weighted_sum_quantile_loss": [
                res["mean_weighted_sum_quantile_loss"][0]
            ],
        }
    )

    print(f"Completed evaluation for {dataset.name}")
    return result_df


def evaluate_datasets(datasets: List[TestDataset], test_results: List[TestResultV2]) -> pd.DataFrame:
    results = []
    model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = torch.compile(model)

    # Process all tasks for this checkpoint
    for dataset in datasets:
        print(f"Evaluating {dataset.name}")
        partition = Partition.TEST
        test_time_profile = TimeProfile(partition.name)
        with TimeProfiler(test_time_profile):
            result_df = evaluate_dataset_with_model(model, dataset)

        if result_df is not None:
            results.append(result_df)

            model_score_metrics = [
                ModelScoreMetrics(
                    MetricType.MAE,
                    partition=partition,
                    score=float(result_df.loc[0, "eval_metrics/MAE[0.5]"]),
                ),
                ModelScoreMetrics(
                    MetricType.MAPE,
                    partition=partition,
                    score=float(result_df.loc[0, "eval_metrics/MAPE[0.5]"]),
                ),
            ]
            model_time_profiles = [
                ModelTimeProfiles(
                    TimeProfileType.TOTAL_CLOCK_TIME,
                    partition,
                    Seconds(test_time_profile.time_ellipse.to_float()/dataset.num_of_forecast_points),
                )
            ]
            test_result = TestResultV2(
                dataset_name=dataset.name,
                model_score_metrics=model_score_metrics,
                model_time_profiles=model_time_profiles,
            )
            test_results.append(test_result)

    # Cleanup model and memory only after completing all tasks
    del model
    torch.cuda.empty_cache()
    gc.collect()

    if not results:
        print("No successful evaluations in this task batch")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


SEED = 42
USE_KV_CACHE = True
NUM_SAMPLES = 256


def run_eval():
    print(f"Evaluating DataRobot MBTest Benchmark")

    # Process all tasks sequentially
    test_results: List[TestResultV2] = []
    datarobot_mbtest_yaml_path = Path(
        "/home/lkanggithub/projects/foundation_model_compare/"
        "custom_data_no_pii_ts_with_lab_machine_path.yaml"
    )
    time_series_frequence = "D"
    test_datasets = TestDataset.create_from_datarobot_mbtest_yaml(
        datarobot_mbtest_yaml_path, time_series_frequence, True,
    )
    evaluate_datasets(test_datasets, test_results)

    TestResultV2.to_csv(test_results, Path("/home/lkanggithub/projects/foundation_model_compare/results_datarobot_mbtest_with_toto_gpu_with_time.csv"))


if __name__ == "__main__":
    run_eval()