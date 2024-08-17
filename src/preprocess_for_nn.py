import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
import preprocess_for_all_models as my_preprocess
import word2vec_for_categorical_col as my_word2vec

# 各種定数
DROP_COLUMNS = ["MonthlyIncome", "customer_info", "customer_info_concat"]
NUMERIC_COLUMNS = [
    "CityTier", "DurationOfPitch", "Occupation",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "NumberOfTrips",
    "Designation", "Age",
    "MonthlyIncome_numeric", "children"
]
HIERARCHICAL_COLUMNS = ["Occupation", "ProductPitched", "Designation"]
CATEGORICAL_COLUMNS = ["TypeofContact", "marriage_history"]

# Ordinal Encodingで使用する順序リスト
OCCUPATION_ORDER = ["Salaried", "Small Business", "Large Business"]
PRODUCT_PITCHED_ORDER = ["basic", "standard", "deluxe", "super deluxe", "king"]
# PRODUCT_PITCHED_ORDER = ["basic", "standard", "deluxe", "king", "super deluxe"]
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
    # # すべてのモデルに共通する前処理を実施
    # train_df, test_df = my_preprocess.preprocess_total(train_df, test_df)

    # # word2vecを使用してカテゴリカル変数をエンベディング
    # train_df, test_df = my_word2vec.word2vec_for_customer_info(
        # train_df,
        # test_df
    # )

    # 不要なカラムを削除
    for col in DROP_COLUMNS:
        try:
            train_df = train_df.drop(columns=col)
            test_df = test_df.drop(columns=col)
        except Exception as drop_error:
            print(drop_error)
            pass

    # -1で欠損値を補完し、新しいデータフレームを作成
    for df in [train_df, test_df]:
        df["TypeofContact"] = df["TypeofContact"].fillna(-1)
        df["MonthlyIncome_numeric"] = df["MonthlyIncome_numeric"].fillna(df["MonthlyIncome_numeric"].median())

    # Ordinal Encodingの実施
    ordinal_encoder = OrdinalEncoder(categories=CATEGORIES_ORDER)

    # 両方のデータフレームをエンコード
    train_df_encoded = ordinal_encoder.fit_transform(train_df[HIERARCHICAL_COLUMNS])
    test_df_encoded = ordinal_encoder.fit_transform(test_df[HIERARCHICAL_COLUMNS])  # それぞれ独自にエンコード

    train_df[HIERARCHICAL_COLUMNS] = train_df_encoded
    test_df[HIERARCHICAL_COLUMNS] = test_df_encoded

    # one-hot encodingの実施
    train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLUMNS, dtype='float32')
    test_df = pd.get_dummies(test_df, columns=CATEGORICAL_COLUMNS, dtype='float32')

    # 数値データのスケーリング
    train_scaler = MinMaxScaler()
    test_scaler = MinMaxScaler()

    train_df[NUMERIC_COLUMNS] = train_scaler.fit_transform(train_df[NUMERIC_COLUMNS])
    test_df[NUMERIC_COLUMNS] = test_scaler.fit_transform(test_df[NUMERIC_COLUMNS])  # それぞれ独自にスケーリング

    # childrenだけなぜか埋まっていないので欠損値を-1で埋める
    for df in [train_df, test_df]:
        df["children"] = df["children"].fillna(-1)

    # 全てのカラムを float32 型に変換
    train_df = train_df.astype('float32')
    test_df = test_df.astype('float32')

    return train_df, test_df
