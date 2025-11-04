import torch
import numpy as np
import pandas as pd
import timesfm

# 모델을 저장할 전역 변수
_CACHED_MODEL = None

def _load_model():
    """Loads the TimesFM model and caches it."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        print("[TimesFM] Loading model for the first time... (This may take a moment)")
        
        torch.set_float32_matmul_precision("high")
        
        # 사용 가능한 디바이스(CUDA, MPS, CPU)를 자동으로 찾습니다.
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        _CACHED_MODEL = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            device=device,
            torch_dtype=torch.bfloat16,
        )
        
        print(f"[TimesFM] Model loaded and compiled successfully on device: {device}")
    return _CACHED_MODEL


def get_timesfm_predictions_data(train, test, num_samples=10, **kwargs):
    """
    Obtain forecasts from Google's TimesFM model using the official timesfm library.
    """
    if isinstance(train, list):
        train_series = train[0]
        test_series = test[0]
    else:
        train_series = train
        test_series = test
        
    # [새로운 디버깅 코드] 데이터 유효성 검사
    if train_series.empty or test_series.empty:
        raise ValueError("입력 데이터 (train/test)가 비어있습니다.")
        
    # TimesFM expects numpy arrays as input
    train_values = train_series.values
    
    # [새로운 디버깅 코드] train_values의 형태 확인
    if len(train_values) == 0:
        raise ValueError("train_values가 비어 있습니다.")
                
    # Use a context length supported by the model, e.g., 1024
    # The model can handle various context lengths, but this is a reasonable default.
    context_length = 1024
    prediction_length = len(test)

    # Truncate the training data to fit the context length
    context_values = train_values[-context_length:]
    
    # Load the pre-trained and compiled model
    model = _load_model()

    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    # The model expects a list of inputs, even if there's only one.
    forecast_inputs = [context_values]

    # Generate point forecasts and quantile forecasts
    point_forecasts, quantile_forecasts = model.forecast(
        inputs=forecast_inputs,
        horizon=prediction_length,
    )

    point_forecasts.shape  # (2, 12)
    quantile_forecasts.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.

    # The output is a list of forecasts, one for each input. We take the first one.
    median_values = point_forecasts[0]
    median_values = median_values.squeeze().flatten()
    
    # To generate samples, we can use the quantile forecasts if available,
    # or sample from a distribution around the point forecast.
    # For simplicity, we'll sample from a normal distribution using residuals' std dev.
    if quantile_forecasts is not None and quantile_forecasts[0].shape[-1] > 1:
        # A more advanced implementation could use the quantiles to create samples.
        # For now, we'll stick to the simpler method for consistency.
        pass

    residuals = train_values[1:] - train_values[:-1]
    sigma = np.std(residuals) if np.std(residuals) > 0 else 0.1 # Avoid zero sigma

    samples = np.random.normal(loc=median_values, scale=sigma, size=(num_samples, prediction_length))
    
    samples_df = pd.DataFrame(samples, columns=test_series.index)
    median_series = pd.Series(median_values, index=test_series.index)
    
    out_dict = {
        'NLL/D': np.nan,
        'samples': samples_df,
        'median': median_series,
        'info': {'Method': 'TimesFM'}
    }
    return out_dict