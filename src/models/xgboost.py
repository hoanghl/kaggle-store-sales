from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm

from src import utils
from src.data import EcuardoSales, transformers

ADF_STATIONARY_P = 0.05

if __name__ == "__main__":
    # reg_catboost = CatBoostRegressor(
    #     n_estimators=200,
    #     loss_function="RMSE",
    #     learning_rate=1e-3,
    #     task_type="CPU",
    #     random_state=37,
    #     verbose=False,
    #     # border_count=128,
    # )
    # reg_xgb = xgb.XGBRegressor(
    #     tree_method="hist",
    #     enable_categorical=True,
    #     n_estimators=1000,
    #     max_depth=7,
    #     eta=0.1,
    #     subsample=0.7,
    # )
    # reg_lgb = lgb.LGBMRegressor(
    #     num_leaves=15,
    #     max_depth=-1,
    #     random_state=37,
    #     silent=True,
    #     metric="rmse",
    #     n_jobs=4,
    #     n_estimators=1000,
    #     colsample_bytree=0.9,
    #     subsample=0.9,
    #     learning_rate=1e-3,
    # )

    # regressor = reg_xgb

    conf = utils.load_conf()
    data = EcuardoSales(conf["PATH"], diff_order=1)

    # path_train = "data/processed/train.csv"
    # path_val = "data/processed/val.csv"
    # df_train_raw = pd.read_csv(path_train, parse_dates=["date"], index_col="id")
    # df_val_raw = pd.read_csv(path_val, parse_dates=["date"], index_col="id")

    # df = pd.concat((df_train_raw, df_val_raw)).sort_values(by="date")
    # df_ = df[(df["family"] == "AUTOMOTIVE") & (df["store_nbr"] == 1)]

    models = {
        family: xgb.XGBRegressor(
            tree_method="hist",
            enable_categorical=True,
            n_estimators=1000,
            max_depth=7,
            eta=0.1,
            subsample=0.7,
        )
        for family in data.df_val_raw["family"].unique()
        # for family in df_["family"].unique()
    }

    with tqdm(total=115926) as pbar:
        for store, family, Xtrain, ytrain, Xval, yval in data.gen_Xy_trainval(deterministic=False):
            # pool_train = Pool(Xtrain, ytrain, cat_features=utils.cat_features)
            # pool_val = Pool(Xval.iloc[1:], yval.iloc[1:], cat_features=utils.cat_features)
            # reg_catboost.fit(pool_train)
            # a = reg_catboost.predict(pool_val)

            regressor = models[family]
            regressor.fit(Xtrain, ytrain)
            a = regressor.predict(Xval.iloc[1:])

            p = data.meta.get_adf_test(store, family)
            has_shift = p > ADF_STATIONARY_P
            pipe_sales = transformers.make_pipeline_sale(has_shift)
            pipe_sales.fit(yval)

            pred = np.clip(pipe_sales.inverse_transform(a[:, None]), 0, None)
            if has_shift is False:
                pred = np.insert(pred, 0, 0, axis=0)

            msle = mean_squared_log_error(yval[1:], pred[1:])

            pbar.set_postfix({"store": store, "family": family, "msle": msle})
            pbar.update(1)

            # break

        # logger.info(f"store: {store} - family: {family} - msle: {msle:.5f}")

    path_models = Path("models/xgboost") / datetime.now().strftime(r"%Y%m%d_%H%M%S")
    path_models.mkdir(exist_ok=True, parents=True)
    for name, model in models.items():
        path_save_model = path_models / f"{name.replace('/', '_')}.json"
        model.save_model(path_save_model)
