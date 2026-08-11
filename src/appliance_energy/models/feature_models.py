from xgboost import XGBRegressor


def create_xgboost_model(random_state=42):
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


def train_feature_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
