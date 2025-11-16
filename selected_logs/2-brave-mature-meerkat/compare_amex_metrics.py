import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple


def _df_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true_df = pd.DataFrame({"target": y_true})
    y_pred_df = pd.DataFrame({"prediction": y_pred})
    return y_true_df, y_pred_df


def amex_metric_pipeline0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    df_cut = df[df["weight_cumsum"] <= four_pct_cutoff]
    top4 = df_cut["target"].sum() / df["target"].sum()
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    weighted_gini = df["gini"].sum()
    df_true = df.copy()
    df_true["prediction"] = df_true["target"]
    df_true = df_true.sort_values("prediction", ascending=False)
    df_true["weight"] = df_true["target"].apply(lambda x: 20 if x == 0 else 1)
    df_true["random"] = (df_true["weight"] / df_true["weight"].sum()).cumsum()
    total_pos2 = (df_true["target"] * df_true["weight"]).sum()
    df_true["cum_pos_found"] = (df_true["target"] * df_true["weight"]).cumsum()
    df_true["lorentz"] = df_true["cum_pos_found"] / total_pos2
    df_true["gini"] = (df_true["lorentz"] - df_true["random"]) * df_true["weight"]
    perfect_gini = df_true["gini"].sum()
    norm_gini = weighted_gini / perfect_gini
    return 0.5 * (norm_gini + top4)


def amex_metric_pipeline1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_df, y_pred_df = _df_inputs(y_true, y_pred)

    def top_four_percent_captured(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["wcum"] = df["weight"].cumsum()
        return (df.loc[df["wcum"] <= cutoff, "target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["rand"] = df["weight"].cumsum() / df["weight"].sum()
        df["cum_pos"] = (df["target"] * df["weight"]).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["lorentz"] = df["cum_pos"] / total_pos
        df["gini"] = (df["lorentz"] - df["rand"]) * df["weight"]
        return df["gini"].sum()

    def norm_gini(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        return weighted_gini(y_true_df, y_pred_df) / weighted_gini(
            y_true_df, y_true_df.rename(columns={"target": "prediction"})
        )

    y_t = y_true_df.reset_index(drop=True)
    y_p = y_pred_df.reset_index(drop=True)
    g = norm_gini(y_t, y_p)
    d = top_four_percent_captured(y_t, y_p)
    return 0.5 * (g + d)


def amex_metric_pipeline2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_df, y_pred_df = _df_inputs(y_true, y_pred)
    df = pd.concat([y_true_df, y_pred_df], axis=1)
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    d = (df.loc[df["weight_cumsum"] <= four_pct_cutoff, "target"] == 1).sum() / (df["target"] == 1).sum()
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    g = df["gini"].sum() / df["gini"].sum()
    return 0.5 * (g + d)


def amex_metric_pipeline3(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    df_g = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_g["random"] = df_g["weight"].cumsum() / df_g["weight"].sum()
    total_pos = (df_g["target"] * df_g["weight"]).sum()
    df_g["cum_pos"] = (df_g["target"] * df_g["weight"]).cumsum()
    df_g["lorentz"] = df_g["cum_pos"] / total_pos
    df_g["gini_contrib"] = (df_g["lorentz"] - df_g["random"]) * df_g["weight"]
    weighted_gini = df_g["gini_contrib"].sum()
    df_perfect = df_g.copy()
    df_perfect = df_perfect.sort_values("target", ascending=False).reset_index(drop=True)
    df_perfect["random"] = df_perfect["weight"].cumsum() / df_perfect["weight"].sum()
    df_perfect["cum_pos"] = (df_perfect["target"] * df_perfect["weight"]).cumsum()
    df_perfect["lorentz"] = df_perfect["cum_pos"] / total_pos
    df_perfect["gini_contrib"] = (df_perfect["lorentz"] - df_perfect["random"]) * df_perfect["weight"]
    perfect_gini = df_perfect["gini_contrib"].sum()
    norm_gini = weighted_gini / perfect_gini
    df_d = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_d["weight_cum"] = df_d["weight"].cumsum()
    cutoff = int(0.04 * df_d["weight"].sum())
    top = df_d.loc[df_d["weight_cum"] <= cutoff]
    d = top["target"].sum() / df_d["target"].sum()
    return 0.5 * (norm_gini + d)


def amex_metric_pipeline4(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_df, y_pred_df = _df_inputs(y_true, y_pred)

    def top_four_percent_captured(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["wcum"] = df["weight"].cumsum()
        df_cut = df[df["wcum"] <= cutoff]
        return (df_cut["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df_true = y_true_df.rename(columns={"target": "prediction"})
        return weighted_gini(y_true_df, y_pred_df) / weighted_gini(y_true_df, df_true)

    g = normalized_weighted_gini(y_true_df, y_pred_df)
    d = top_four_percent_captured(y_true_df, y_pred_df)
    return 0.5 * (g + d)


def amex_metric_pipeline5(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return amex_metric_pipeline4(y_true, y_pred)


def amex_metric_pipeline6(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_df, y_pred_df = _df_inputs(y_true, y_pred)
    y_true_copy = y_true_df.rename(columns={"target": "prediction"})

    def top_four_percent_captured(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        top = df.loc[df["weight_cumsum"] <= cutoff]
        return (top["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
        df = pd.concat([y_true_df, y_pred_df], axis=1).sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    g = weighted_gini(y_true_df, y_pred_df) / weighted_gini(y_true_df, y_true_copy)
    d = top_four_percent_captured(y_true_df, y_pred_df)
    return 0.5 * (g + d)


def amex_metric_pipeline7(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return amex_metric_pipeline4(y_true, y_pred)


def amex_metric_pipeline8(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return amex_metric_pipeline4(y_true, y_pred)


def amex_metric_pipeline9(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return amex_metric_pipeline4(y_true, y_pred)


TEST_CASES: List[Tuple[str, np.ndarray, np.ndarray]] = [
    (
        "Case A",
        np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0]),
        np.array([0.05, 0.10, 0.90, 0.40, 0.65, 0.80, 0.15, 0.30, 0.70, 0.20]),
    ),
    (
        "Case B",
        np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]),
        np.array([0.95, 0.20, 0.10, 0.85, 0.60, 0.40, 0.55, 0.35, 0.30, 0.25, 0.15, 0.05]),
    ),
]


FUNCTIONS: List[Tuple[str, Callable[[np.ndarray, np.ndarray], float]]] = [
    (f"pipeline{i}", globals()[f"amex_metric_pipeline{i}"]) for i in range(10)
]


def main() -> None:
    tolerance = 1e-9
    for case_name, y_true, y_pred in TEST_CASES:
        print(f"\n=== {case_name} ===")
        results: Dict[str, float] = {}
        for name, func in FUNCTIONS:
            score = func(y_true, y_pred)
            results[name] = score
            print(f"{name}: {score:.12f}")
        baseline = next(iter(results.values()))
        mismatched = [
            name for name, score in results.items() if abs(score - baseline) > tolerance
        ]
        if mismatched:
            print("Mismatch vs baseline:", ", ".join(mismatched))
        else:
            print("All pipelines match within tolerance.")


if __name__ == "__main__":
    main()

