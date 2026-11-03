"""任务34：train_svd_model。保留原始分析口径与输出文件名。"""

def run():
    from amazon_project.common import ROOT as PROJECT_ROOT
    import json
    import time
    from importlib.metadata import version

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from surprise import Dataset, Reader, SVD, dump


    ROOT = PROJECT_ROOT
    DATA_DIR = ROOT / "data_clean" / "week3"
    MODEL_DIR = ROOT / "models" / "week3"
    TABLE_DIR = ROOT / "reports" / "week3" / "tables"
    FIGURE_DIR = ROOT / "reports" / "week3" / "figures"

    # 本次预先固定参数，不根据测试集结果反复调整。
    PARAMETERS = {
        "n_factors": 50,
        "n_epochs": 20,
        "biased": True,
        "lr_all": 0.005,
        "reg_all": 0.02,
        "random_state": 42,
    }


    def read_ratings(path):
        return pd.read_csv(
            path,
            usecols=[
                "source_row", "user_id",
                "product_id", "rating",
            ],
            dtype={
                "source_row": "int64",
                "user_id": "string",
                "product_id": "string",
                "rating": "float64",
            },
        )


    def main():
        for folder in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
            folder.mkdir(parents=True, exist_ok=True)

        train = read_ratings(DATA_DIR / "recommendation_train.csv")
        test = read_ratings(DATA_DIR / "recommendation_test.csv")

        if train.empty or test.empty:
            raise ValueError("训练集或测试集为空。")

        for name, frame in [("train", train), ("test", test)]:
            if frame.isna().any().any():
                raise ValueError(f"{name}存在缺失值。")
            if not frame["rating"].between(1, 5).all():
                raise ValueError(f"{name}存在异常评分。")
            if frame.duplicated(["user_id", "product_id"]).any():
                raise ValueError(f"{name}存在重复用户—商品组合。")

        overlap = train[["user_id", "product_id"]].merge(
            test[["user_id", "product_id"]],
            on=["user_id", "product_id"],
            how="inner",
        )
        if not overlap.empty:
            raise ValueError("训练集与测试集存在重叠组合。")

        if not test["user_id"].isin(set(train["user_id"])).all():
            raise ValueError("测试集中出现未在训练集出现的用户。")

        seen_mask = test["product_id"].isin(
            set(train["product_id"])
        ).to_numpy()

        print("=== 输入数据 ===")
        print(f"训练记录数：{len(train):,}")
        print(f"测试记录数：{len(test):,}")

        # 基准方法的平均分只能使用训练集计算。
        global_mean = float(train["rating"].mean())
        user_means = train.groupby("user_id")["rating"].mean()

        result = test.copy()
        result["product_seen_in_train"] = seen_mask
        result["pred_GlobalMean"] = global_mean
        result["pred_UserMean"] = (
            test["user_id"].map(user_means).to_numpy(dtype=float)
        )

        print("\n开始训练SVD模型……", flush=True)

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            train[["user_id", "product_id", "rating"]],
            reader,
        )
        trainset = dataset.build_full_trainset()

        model = SVD(**PARAMETERS, verbose=True)

        start = time.perf_counter()
        model.fit(trainset)
        training_seconds = time.perf_counter() - start

        print("\n训练完成，开始预测测试集……", flush=True)

        # 预测时不向模型提供测试集真实评分。
        predictions = []
        for number, (user_id, product_id) in enumerate(
            test[["user_id", "product_id"]].itertuples(
                index=False, name=None
            ),
            start=1,
        ):
            prediction = model.predict(
                str(user_id),
                str(product_id),
                clip=True,
            )
            predictions.append(float(prediction.est))

            if number % 20_000 == 0:
                print(f"已预测 {number:,} 条", flush=True)

        result["pred_SVD"] = predictions

        prediction_columns = [
            "pred_GlobalMean", "pred_UserMean", "pred_SVD"
        ]
        if not np.isfinite(
            result[prediction_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError("预测结果出现缺失或非有限数值。")

        # 保存逐条预测到本地，后续计算推荐指标时使用。
        result.to_csv(
            DATA_DIR / "recommendation_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 模型也保存在本地，之后不必重复训练。
        model_path = MODEL_DIR / "recommendation_svd.model"
        dump.dump(str(model_path), algo=model)

        metric_rows = []
        groups = {
            "all_test": np.ones(len(result), dtype=bool),
            "seen_product": seen_mask,
            "unseen_product": ~seen_mask,
        }

        for group_name, mask in groups.items():
            part = result.loc[mask]
            if part.empty:
                continue

            actual = part["rating"].to_numpy(dtype=float)

            for model_name in ["GlobalMean", "UserMean", "SVD"]:
                estimated = part[
                    f"pred_{model_name}"
                ].to_numpy(dtype=float)
                errors = estimated - actual

                metric_rows.append({
                    "evaluation_group": group_name,
                    "model": model_name,
                    "test_rows": len(part),
                    "MAE": float(np.mean(np.abs(errors))),
                    "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                })

        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(
            TABLE_DIR / "recommendation_rating_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        run_summary = pd.DataFrame([{
            "train_rows": len(train),
            "test_rows": len(test),
            "train_users": train["user_id"].nunique(),
            "train_products": train["product_id"].nunique(),
            "training_global_mean": global_mean,
            "svd_training_seconds": training_seconds,
            "n_factors": PARAMETERS["n_factors"],
            "n_epochs": PARAMETERS["n_epochs"],
            "random_seed": PARAMETERS["random_state"],
        }])
        run_summary.to_csv(
            TABLE_DIR / "recommendation_training_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        config = {
            "algorithm": "Surprise SVD",
            "parameters": PARAMETERS,
            "rating_scale": [1, 5],
            "baselines": ["GlobalMean", "UserMean"],
            "baseline_statistics_source": "training_data_only",
            "test_predictions_clipped_to_rating_scale": True,
            "test_used_for_parameter_tuning": False,
            "evaluation_groups": list(groups),
            "versions": {
                name: version(name)
                for name in [
                    "scikit-surprise", "pandas", "numpy", "matplotlib"
                ]
            },
        }
        with open(
            TABLE_DIR / "recommendation_training_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        # 比较全部测试记录上的预测误差。
        all_metrics = (
            metrics[metrics["evaluation_group"] == "all_test"]
            .set_index("model")
            .reindex(["GlobalMean", "UserMean", "SVD"])
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, metric in zip(axes, ["MAE", "RMSE"]):
            bars = ax.bar(
                all_metrics.index,
                all_metrics[metric],
                color=["#AAB2BD", "#7195B5", "#398675"],
            )
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set_title(f"{metric} on all test ratings")
            ax.set_ylabel("Rating error (lower is better)")
            ax.set_ylim(0, all_metrics[metric].max() * 1.20)

        fig.tight_layout()
        figure_path = (
            FIGURE_DIR / "04_recommendation_rating_errors.png"
        )
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        print("\n=== 评分预测误差 ===")
        print(metrics.round(4).to_string(index=False))

        print(f"\nSVD训练用时：{training_seconds:.1f}秒")
        print(f"训练集平均评分：{global_mean:.4f}")

        print("\n模型：", model_path)
        print(
            "本地逐条预测：",
            DATA_DIR / "recommendation_test_predictions.csv",
        )
        print("统计表：", TABLE_DIR)
        print("图片：", figure_path)
        print(
            "\n完成。本步评价评分预测误差，"
            "尚未计算推荐精确率和召回率。"
        )
        print("训练集、测试集及原始数据没有修改。")


    main()


if __name__ == "__main__":
    run()
