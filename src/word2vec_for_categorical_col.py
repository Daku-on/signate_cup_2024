import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import Optional


def add_tfidf_features(
    df: pd.DataFrame,
    col_name: str,
    max_features: int = 40,
    n_components: Optional[int] = None,
) -> pd.DataFrame:
    """
    与えられたテキスト列をTF-IDFベクトルに変換し、SVDを使用して次元削減を行い、元のデータフレームに追加する関数。
    n_componentsがNoneの場合、次元削減は行わない。

    Args:
        df: 入力データフレーム
        col_name: TF-IDFベクトル化するテキスト列の名前
        max_features: TF-IDFベクトル化時の最大特徴量数
        n_components: SVDによる次元削減後の成分数。Noneの場合、次元削減を行わない。

    Returns:
        df: 変換後の特徴量が追加されたデータフレーム
    """

    # TF-IDFベクトル化のためのVectorizerを初期化
    vectorizer = TfidfVectorizer(max_features=max_features)

    # テキスト列をTF-IDFベクトルに変換
    vectors = vectorizer.fit_transform(df[col_name])

    if n_components is not None:
        # SVDを使用して次元削減
        svd = TruncatedSVD(n_components=n_components)
        reduced_vectors = svd.fit_transform(vectors)
        # ベクトルの正規化
        normalized_vectors = normalize(reduced_vectors, norm="l2")
        # データフレームに変換
        tfidf_df = pd.DataFrame(normalized_vectors, index=df.index)
        # 新しいデータフレームの列名を設定（次元削減後）
        cols = [(col_name + "_svd_" + str(f)) for f in range(tfidf_df.shape[1])]
        tfidf_df.columns = cols
    else:
        # 次元削減を行わない場合、TF-IDFベクトルをデータフレームに変換
        tfidf_df = pd.DataFrame(vectors.toarray(), index=df.index)
        # 新しいデータフレームの列名を設定（次元削減なし）
        cols = [(col_name + "_tfidf_" + str(f)) for f in range(tfidf_df.shape[1])]
        tfidf_df.columns = cols

    # 変換された特徴量を元のデータに結合
    df = pd.concat([df, tfidf_df], axis="columns")

    return df


def concatenate_columns(
    df: pd.DataFrame,
    columns: list[str],
    new_col_name: str,
) -> pd.DataFrame:
    """
    複数のカラムをアンダースコアで結合し、新しいカラムとして追加する関数

    Args:
        df: 入力データフレーム
        columns: 結合するカラムのリスト
        new_col_name: 新しいカラム名

    Returns:
        df: 新しいカラムが追加されたデータフレーム
    """

    # 複数のカラムをアンダースコアで結合
    df[new_col_name] = df[columns].astype(str).agg('_'.join, axis=1)

    return df


def word2vec_for_customer_info(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    use after preprocessing_for_all_models.py
    """

    train_df = concatenate_columns(
        train_df,
        ["marriage_history", "car", "children"],
        "customer_info_concat"
    )
    test_df = concatenate_columns(
        test_df,
        ["marriage_history", "car", "children"],
        "customer_info_concat"
    )
    train_df = add_tfidf_features(
        train_df,
        "customer_info_concat",
        max_features=40,  # I know 40 is at most for this case
        n_components=n_components
    )
    test_df = add_tfidf_features(
        test_df,
        "customer_info_concat",
        max_features=40,
        n_components=n_components
    )

    return train_df, test_df
