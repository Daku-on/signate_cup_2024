# ライブラリのimportを行います
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
    col_list = df_train.columns
    df_processed = df_processed[col_list]

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