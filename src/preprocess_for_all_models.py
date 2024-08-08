# ライブラリのimportを行います
import re
import pandas as pd
import numpy as np
import unicodedata


# kkou27  担当分

# 「Age」に対する処理
unit_list = ['歳', '才', '際', '代']
def remove_specific_chars(text):
    for char in unit_list:
        text = text.replace(char, '')
    return text


def kanji_to_number(kanji):
    # 漢数字をアラビア数字に変換する辞書
    kanji_to_num = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
        '十': 10, '百': 100, '千': 1000, '万': 10000,
        '億': 100000000, '兆': 1000000000000
    }
    result = 0
    tmp = 0
    base = 1
    for char in reversed(kanji):
        if char in kanji_to_num:
            num = kanji_to_num[char]
            if num >= 10:
                if tmp == 0:
                    tmp = 1
                result += tmp * num
                tmp = 0
            else:
                tmp += num * base
                base = 1
        else:
            return kanji  # 漢数字でなければそのまま返す
    if tmp != 0:
        result += tmp
    return result


def Age_transform(df_train, df_test, target = 'train'):
    """
    Ageを処理する関数
    Arges:
    - df_train: 訓練データ
    - df_test: テストデータ
    - target: 訓練データとテストデータのうち、処理の対象とするデータを決めるparameter
    """
    df_train_temp = df_train.copy()
    df_test_temp = df_test.copy()
    # 特定の文字を取り除いた新しいカラムを作成
    df_train_temp['Age_number'] = df_train_temp['Age'].astype(str).apply(remove_specific_chars)
    # nanを-1に置き換え
    df_train_temp['Age_number'] = df_train_temp['Age_number'].replace('nan', '-1')
    # 単位カラムを作成
    df_train_temp['Age_unit'] = df_train_temp['Age'].astype(str).apply(lambda x: ''.join([char for char in x if char in unit_list]))
    # 漢数字を数字に変換
    df_train_temp['Age_number']  = df_train_temp['Age_number'].apply(kanji_to_number)
    # 数値型に変換
    df_train_temp['Age_number'] = df_train_temp['Age_number'].astype(float)
    df_train_temp['Age_number'] = df_train_temp['Age_number'].astype(int)

    # 特定の文字を取り除いた新しいカラムを作成
    df_test_temp['Age_number'] = df_test_temp['Age'].astype(str).apply(remove_specific_chars)
    # nanを-1に置き換え
    df_test_temp['Age_number'] = df_test_temp['Age_number'].replace('nan', '-1')
    # 単位カラムを作成
    df_test_temp['Age_unit'] = df_test_temp['Age'].astype(str).apply(lambda x: ''.join([char for char in x if char in unit_list]))
    # 漢数字を数字に変換
    df_test_temp['Age_number']  = df_test_temp['Age_number'].apply(kanji_to_number)
    # 数値型に変換
    df_test_temp['Age_number'] = df_test_temp['Age_number'].astype(float)
    df_test_temp['Age_number'] = df_test_temp['Age_number'].astype(int)

    df_all = pd.concat([df_train_temp, df_test_temp], axis=0)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-9歳', '10-19歳', '20-29歳', '30-39歳', '40-49歳',
                '50-59歳', '60-69歳', '70-79歳', '80-89歳', '90-100歳']

    # 年代ごとのカテゴリを作成
    df_all['年代'] = pd.cut(df_all['Age_number'], bins=bins, labels=labels, right=False)

    median_ages = df_all.groupby('年代',observed=False)['Age_number'].median()
    df_median = median_ages.reset_index()
    df_median.rename(columns={'Age_number': 'median_age'}, inplace=True)

    if target == 'train':
        # 年代カラムを作成
        df_train_temp['年代'] = pd.cut(df_train_temp['Age_number'], bins=bins, labels=labels, right=False)
        # 中央値を結合
        df_train_median = pd.merge(df_train_temp, df_median, on='年代', how='left')
        # 単位が「代」となっている場合はその年代の中央値をAge_numberとする
        df_train_median['Age_number'] = np.where(df_train_median['Age_unit'] == '代', df_train_median['median_age'], df_train_median['Age_number'])
        df_train_median.drop(columns=['Age','Age_unit','年代','median_age'], inplace=True)
        df_train_median.rename(columns={'Age_number': 'Age'}, inplace = True)
        return df_train_median
    else:
        # 年代カラムを作成
        df_test_temp['年代'] = pd.cut(df_test_temp['Age_number'], bins=bins, labels=labels, right=False)
        # 中央値を結合
        df_test_median = pd.merge(df_test_temp, df_median, on='年代', how='left')
        # 単位が「代」となっている場合はその年代の中央値をAge_numberとする
        df_test_median['Age_number'] = np.where(df_test_median['Age_unit'] == '代', df_test_median['median_age'], df_test_median['Age_number'])
        df_test_median.drop(columns=['Age','Age_unit','年代','median_age'], inplace=True)
        df_test_median.rename(columns={'Age_number': 'Age'}, inplace = True)
        return df_test_median


#-------------------------------------------------------------------------------------------------------------------------------------------
# 「TypeofContact」に対する処理
def TypeofContact_transform(df):
    """
    TypeofContactを処理する関数
     Arges:
      - df_train: 前処理をするデータフレーム
     """
    # TypeofContactのnanをunansweredに置き換える
    df['TypeofContact'] = df['TypeofContact'].fillna('unknown')

    return df


#-------------------------------------------------------------------------------------------------------------------------------------------
# 「'DurationOfPitch'」に対する処理
def convert_to_seconds(value):
    if pd.isna(value):
        return -1
    elif '秒' in value:
        return int(value.replace('秒', ''))
    elif '分' in value:
        return int(value.replace('分', '')) * 60
    return -1

def DurationOfPitch_transform(df):
    """
    DurationOfPitchを処理する関数
    Arges:
     - df: 前処理をするデータフレーム
    """

    # 秒単位に統一し、数値型に変換
    df['DurationOfPitch'] = df['DurationOfPitch'].apply(convert_to_seconds)
    df['DurationOfPitch'] = df['DurationOfPitch'].astype(int)

    return df


#-------------------------------------------------------------------------------------------------------------------------------------------
# 「Gender」に対する処理
def normalize_gender(value):
    # 全角を半角に変換し、小文字に統一
    value = unicodedata.normalize('NFKC', value).lower()
    # 余分な空白を削除
    value = value.replace(' ', '')
    return value

def Gender_transform(df):
    """
    Genderを処理する関数
    Arges:
     - df: 前処理をするデータフレーム
    """
    # 前処理の適用
    df['Gender'] = df['Gender'].apply(normalize_gender)

    return df


#-------------------------------------------------------------------------------------------------------------------------------------------
# ラッパー関数
def preprocess_data(df_train,df_test,target = 'train'):
    """
    データの前処理を行う関数
    Arges:
    - df_train: 訓練データ
    - df_test: テストデータ
    - target: 訓練データとテストデータのうち、処理の対象とするデータを決めるparameter

    """
    if target == 'train':
        df_processed = Age_transform(df_train,df_test,target = 'train')
    else:
        df_processed = Age_transform(df_train,df_test,target = 'test')
    df_processed = TypeofContact_transform(df_processed)
    df_processed = DurationOfPitch_transform(df_processed)
    df_processed = Gender_transform(df_processed)

    # df_trainのカラムの並びに揃える
    # col_list = df_train.columns.tolist()
    # # col_list.remove("ProdTaken")
    # # col_list.append("ProdTaken")
    # df_processed = df_processed[col_list]

    return df_processed

# kkou27  担当分終了

# ynb0123  担当分
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    dfのGenderからPitchSatisfactionScoreまでの前処理
    - Gender
        - 名寄せ
        - maleなら0、femaleなら1にする
        - カラム名をGender(is_male)に変更
    - NumberOfPersonVisiting
        - 変換不要
    - NumberOfFollowups
        - nanを0に変換（0が値として存在していないため、0の時にはnanが入ると仮定している）
        - 100, 200, ..., 600を1, 2, ..., 6に変更（%がついて100倍されていると仮定している）
    - ProductPitched
        - 全て小文字に変換
        - '|'をlに変換
        - アルファベットでない文字をアルファベットに変換
        - 全角を半角に変換
    - PreferredPropertyStar
        - 変換不要
    - NumberOfTrips
        - 「年4回」「5」「'3'」など、intとstrが混ざっているため、全てintに変換
    - Passport
        - 変換不要
    - PitchSatisfactionScore
        - 変換不要
    
    Parameters
    ---------------
    df: pd.DataFrame
        学習データ or テストデータ or 学習＋テストデータ
    
    Returns
    ---------------
    df_preprocessed: pd.DataFrame
        前処理済dataframe
    """
    # --------------- 準備 ---------------

    # 入力のdataframeが加工されないようにコピーを取る
    df_preprocessed = df.copy()
    # 全角を半角にする関数
    def zenkaku2hankaku(text):
        return unicodedata.normalize('NFKC', text)

    # --------------- Gender ---------------

    # 全角を半角に
    df_preprocessed['Gender'] = df_preprocessed['Gender'].apply(zenkaku2hankaku)
    # 大文字を小文字に
    df_preprocessed['Gender'] = df_preprocessed['Gender'].str.lower()
    # 男性なら0、女性なら1とする
    df_preprocessed['Gender'] = np.where(df_preprocessed['Gender'].str.contains('f'), 0, 1)
    # カラム名をGender(is_male)に変更
    df_preprocessed = df_preprocessed.rename(columns={'Gender': 'Gender(is_male)'})

    # --------------- NumberOfFollowups ---------------

    df_preprocessed['NumberOfFollowups'] = df_preprocessed['NumberOfFollowups'].replace(
        {
            np.nan: 0,
            100: 1,
            200: 2,
            300: 3,
            400: 4,
            500: 5,
            600: 6
        }
    )

    # --------------- ProductPitched ---------------

    # 変な文字をアルファベットに
    conv2alphabet_dict = {
        'α': 'a',
        'Α': 'a',
        'в': 'b',
        'β': 'b',
        '𐊡': 'b',
        'ς': 'c',
        'ϲ': 'c',
        'с': 'c',
        '𝔡': 'd',
        'ᗞ': 'd',
        'ꭰ': 'd',
        'ε': 'e',
        'ı': 'i',
        '|': 'l',
        'ո': 'n',
        'տ': 's',
        'ꓢ': 's',
        'ѕ': 's',
        '×': 'x'
    }

    def conv2alphabet(text, replacements):
        t = str.maketrans(replacements)
        return text.translate(t)
    df_preprocessed['ProductPitched'] = df_preprocessed['ProductPitched'].apply(
        lambda x: conv2alphabet(x, conv2alphabet_dict)
    )
    # 全角を半角に
    df_preprocessed['ProductPitched'] = df_preprocessed['ProductPitched'].apply(zenkaku2hankaku)
    # 大文字を小文字に
    df_preprocessed['ProductPitched'] = df_preprocessed['ProductPitched'].str.lower()
    # なぜか変換できないものたちをパワーで変換
    conv_dict = {
        'вasic': 'basic',
        'ѕuper deluxe': 'super deluxe',
        'baտic': 'basic',
        'ꭰeluxe': 'deluxe',
        'βasic': 'basic',
        'տuper deluxe': 'super deluxe',
        'տtandard': 'standard',
        'standarꭰ': 'standard',
        'basiс': 'basic',
        'dεluxε': 'deluxe',
        'basιc': 'basic',
        'super ꭰeluxe': 'super deluxe',
        'deluxε': 'deluxe',
        'ѕtandard': 'standard',
        'super dεluxe': 'super deluxe',
        'βasiс': 'basic',
        'supεr ꭰeluxe': 'super deluxe',
        'basιс': 'basic',
        'baѕic': 'basic'
    }
    df_preprocessed['ProductPitched'] = df_preprocessed['ProductPitched'].replace(conv_dict)

    # --------------- NumberOfTrips ---------------

    # 日本語をintに
    conv2ntrips = {
        '年に1回': 1,
        '年に2回': 2,
        '年に3回': 3,
        '年に4回': 4,
        '年に5回': 5,
        '年に6回': 6,
        '年に7回': 7,
        '年に8回': 8,
        '半年に1回': 2,
        '四半期に1回': 4
    }
    df_preprocessed['NumberOfTrips'] = df_preprocessed['NumberOfTrips'].replace(conv2ntrips)
    # nanをintに
    df_preprocessed['NumberOfTrips'] = df_preprocessed['NumberOfTrips'].replace({np.nan: 0})
    # strをintに
    df_preprocessed['NumberOfTrips'] = df_preprocessed['NumberOfTrips'].astype(int)

    return df_preprocessed

# ynb0123  担当分終了

# Daku-on  担当分


def convert_fullwidth_to_halfwidth_and_extract_invalid(
    df: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """
    指定されたカラムに対して次の処理を行う:
    1. 全角英字を半角英字に変換
    2. それでもなおAからz以外の文字が含まれるユニークな値をリストとして返す

    Args:
        df (pd.DataFrame): 対象のデータフレーム
        column_name (str): 操作を行うカラムの名前

    Returns:
        pd.DataFrame: 修正後のデータフレーム
        list: 条件に合わないレコードのユニークな値
    """
    # 全角英字を半角英字に変換
    def convert_fullwidth_to_halfwidth(text: str) -> str:
        """
        全角英字を半角英字に変換するヘルパー関数

        Args:
            text (str): 入力文字列

        Returns:
            str: 半角に変換された文字列
        """
        return "".join(
            chr(ord(char) - 65248) if "Ａ" <= char <= "Ｚ" or "ａ" <= char <= "ｚ" else char
            for char in text
        )

    # 対象カラムの全角英字を半角英字に変換
    df[column_name] = df[column_name].apply(
        lambda x: convert_fullwidth_to_halfwidth(x) if pd.notna(x) else x
    )

    # 半角スペースをアンダースコアに置換
    df[column_name] = df[column_name].str.replace(" ", "_", regex=False)

    replace_dict = {
        "𝙧": "r",
        "α": "a",
        "Տ": "S",
        "ѵ": "v",
        "×": "x",
        "е": "e",
        "Α": "A",
        "А": "A",
        "Μ": "M",
        "Е": "E",
        "Ѕ": "S",
    }
    df = df.replace(
        {column_name: replace_dict},
        regex=True
    )

    # Aからz以外の文字を含む行のindexを取得
    invalid_indices = df[~df[column_name].str.match(r"^[A-Za-z_]+$", na=False)].index

    if len(invalid_indices) == 0:
        # ここまでで前処理が完了していれば、カラムの値を小文字に変換して返す
        df[column_name] = df[column_name].apply(
            lambda x: x.lower() if pd.notna(x) else x
        )
        return df, []
    else:
        # 条件に合わないレコードのユニークな値をリストとして取得
        print("there are invalid values in the column: {}".format(column_name))
        unique_invalid_values = df.loc[invalid_indices, column_name].unique().tolist()
        return df, unique_invalid_values


def extract_and_convert_to_numeric(
    df: pd.DataFrame,
    column_name: str,
    new_column_name: str
) -> pd.DataFrame:
    """
    指定されたカラムから数字と「万」を抽出し、1万倍して新しいカラムに保存する。
    正規表現にマッチしない場合、そのインデックスと値を記録する。

    Args:
        df (pd.DataFrame): 入力データフレーム
        column_name (str): 元のカラム名
        new_column_name (str): 結果を保存する新しいカラム名

    Returns:
        pd.DataFrame: 処理結果が保存された新しいカラムが追加されたデータフレーム
        list: 正規表現にマッチしなかったユニークな値のリスト
    """
    unmatched_values = []

    def convert_to_number(
        text: object, index: int
    ) -> int:
        if text is None or pd.isna(text):
            return np.nan
        text = str(text)
        # 正規表現で「万」と数字を含む部分を抽出
        match = re.search(r"(\d+(\.\d+)?)(万)?", text)
        if not match:
            unmatched_values.append((index, text))
            return None
        number_str, _, unit = match.groups()
        number = float(number_str)
        if unit == "万":
            number *= 10000
        return int(number)

    # 各レコードに対して処理を行い、新しいカラムに保存
    df[new_column_name] = [
        convert_to_number(value, idx) 
        for idx, value in enumerate(df[column_name])
    ]
    df[new_column_name] = df[new_column_name].astype(np.float32)

    if len(unmatched_values) == 0:
        return df, []
    else:
        # 正規表現にマッチしなかったユニークな値をリストにして返す
        print("there are unmatched values in the column: {}".format(column_name))
        unique_unmatched_values = list({value for _, value in unmatched_values})
        print(unique_unmatched_values)

        return df, unique_unmatched_values


def customer_info_preprocess(
    df: pd.DataFrame,  # 入力のデータフレーム
    column_name: str,  # 処理対象のカラム名
) -> pd.DataFrame:
    """
    各レコードに対して、指定されたカラムの文字列を処理し、
    各単語を条件に基づいて新しいカラムに分類する。
    Args:
        df (pd.DataFrame): 処理対象のデータフレーム
        column_name (str): 対象カラム名
    Returns:
        pd.DataFrame: 処理結果を含むデータフレーム
    """
    # 各レコードを処理
    for index, row in df.iterrows():
        # 句読点やコロン、改行などを半角スペースに変換
        cleaned_text = re.sub(r"[、。・：；,.;:?!/／\n]", " ", str(row[column_name]))

        # 単語に分割
        words = cleaned_text.split()

        # 各単語に対して処理を実施
        marriage_history = " ".join([word for word in words if "婚" in word or "独" in word])
        car = " ".join([word for word in words if "車" in word])
        children = " ".join([word for word in words if "婚" not in word and "独" not in word and "車" not in word])

        # 各レコードに新しいカラムを追加
        df.at[index, "marriage_history"] = marriage_history
        df.at[index, "car"] = car
        df.at[index, "children"] = children

    # 各カラムの表記揺れを修正
    def dict_replace_function(
        text: str,
        replace_dict: dict
    ) -> str:
        if text in replace_dict:
            return str(replace_dict[text])
        else:
            raise ValueError(f"'{text}' is not found in the replacement dictionary.")

    # car, childrenカラムの各レコードに対して置き換え処理を実施
    # car辞書の作成
    car_replace_dict = {
        "車未所持": 0,
        "自動車未所有": 0,
        "車保有なし": 0,
        "乗用車なし": 0,
        "自家用車なし": 0,
        "車なし": 0,
        "車あり": 1,
        "車所持": 1,
        "自家用車あり": 1,
        "車保有": 1,
        "乗用車所持": 1,
        "自動車所有": 1,
    }
    # children辞書の作成
    children_replace_dict = {
        "子供なし": 0,
        "子供無し": 0,
        "無子": 0,
        "子供ゼロ": 0,
        "非育児家庭": 0,
        "子育て状況不明": np.nan,
        "子の数不詳": np.nan,
        "子供の数不明": np.nan,
        "こども1人": 1,
        "1児": 1,
        "子供1人": 1,
        "子供有り(1人)": 1,
        "子供有り 1人": 1,
        "こども2人": 2,
        "2児": 2,
        "子供2人": 2,
        "子供有り(2人)": 2,
        "こども3人": 3,
        "3児": 3,
        "子供3人": 3,
        "子供有り 2人": 2,
        "子供有り 3人": 3,
        "子供有り(3人)": 3,
        "わからない": np.nan,
        "不明": np.nan,
    }

    # データフレームの対象カラムに適用
    df["car"] = df["car"].apply(dict_replace_function, replace_dict=car_replace_dict)
    df["children"] = df["children"].apply(dict_replace_function, replace_dict=children_replace_dict)

    return df


def preprocess_for_last_3_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    データフレームに対して前処理を行う。

    Args:
        df (pd.DataFrame): 前処理を行うデータフレーム

    Returns:
        pd.DataFrame: 前処理後のデータフレーム
    """

    # カラムごとの前処理
    df, invalid_values = convert_fullwidth_to_halfwidth_and_extract_invalid(df, "Designation")
    df, unmatched_values = extract_and_convert_to_numeric(df, "MonthlyIncome", "MonthlyIncome_numeric")
    df = customer_info_preprocess(df, "customer_info")

    return df

# Daku-on  担当分終了


def preprocess_total(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    dfの前処理
    """
    train_df = preprocess_data(
        train_df,
        test_df,
        target="train"
    )
    test_df = preprocess_data(
        train_df,
        test_df,
        target="test"
    )
    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_df = preprocess_for_last_3_cols(train_df)
    test_df = preprocess_for_last_3_cols(test_df)

    return train_df, test_df
