# ライブラリのimportを行います
import pandas as pd
import numpy as np
import unicodedata


# kkou27  担当分
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


def Age_transform(df_train, df_test, target='train'):
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

# kkou27  担当分終了

# ynb0123  担当分
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

# ynb0123  担当分終了