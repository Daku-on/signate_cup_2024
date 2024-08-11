import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import preprocess_for_all_models as my_preprocess

# 各種定数
DROP_COLUMNS = ["id", "MonthlyIncome", "customer_info"]
NUMERIC_COLUMNS = [
    "CityTier", "DurationOfPitch", "Occupation",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "NumberOfTrips",
    "Designation", "Age",
    "MonthlyIncome_numeric", "children"
]
HIERARCHICAL_COLUMNS = ["Occupation", "ProductPitched", "Designation"]
CATEGORICAL_COLUMNS = ["marriage_history"]

# Ordinal Encodingで使用する順序リスト
OCCUPATION_ORDER = ["Salaried", "Small Business", "Large Business"]
PRODUCT_PITCHED_ORDER = ["basic", "standard", "deluxe", "super deluxe", "king"]
DESIGNATION_ORDER = ["executive", "manager", "senior_manager", "avp", "vp"]
CATEGORIES_ORDER = [OCCUPATION_ORDER, PRODUCT_PITCHED_ORDER, DESIGNATION_ORDER]


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    train_dfとtest_dfに対して同じ前処理を行い、加工済みのデータフレームを返す。

    Args:
        train_df (pd.DataFrame): 加工前の学習データ
        test_df (pd.DataFrame): 加工前のテストデータ

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 加工済みの学習データとテストデータ
    """
    # すべてのモデルに共通する前処理を実施
    train_df, test_df = my_preprocess.preprocess_total(train_df, test_df)

    # 不要なカラムを削除
    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)

    # -1で欠損値を補完し、新しいデータフレームを作成
    for df in [train_df, test_df]:
        df["children"] = df["children"].fillna(-1)
        df["MonthlyIncome_numeric"] = df["MonthlyIncome_numeric"].fillna(df["MonthlyIncome_numeric"].median())

    # Ordinal Encodingの実施
    ordinal_encoder = OrdinalEncoder(categories=CATEGORIES_ORDER)

    # 両方のデータフレームをエンコード
    train_df_encoded = ordinal_encoder.fit_transform(train_df[HIERARCHICAL_COLUMNS])
    test_df_encoded = ordinal_encoder.fit_transform(test_df[HIERARCHICAL_COLUMNS])  # それぞれ独自にエンコード

    train_df[HIERARCHICAL_COLUMNS] = train_df_encoded
    test_df[HIERARCHICAL_COLUMNS] = test_df_encoded

    # one-hot encodingの実施
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLUMNS)
    test_df = pd.get_dummies(test_df, columns=CATEGORICAL_COLUMNS)

    # 数値データのスケーリング
    train_scaler = StandardScaler()
    test_scaler = StandardScaler()

    train_df[NUMERIC_COLUMNS] = train_scaler.fit_transform(train_df[NUMERIC_COLUMNS])
    test_df[NUMERIC_COLUMNS] = test_scaler.fit_transform(test_df[NUMERIC_COLUMNS])  # それぞれ独自にスケーリング

    return train_df, test_df
