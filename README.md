# signate_cup_2024
https://signate.jp/competitions/1376

Signate Cup2024用レポジトリ (終了後公開)

[Kaggle日記という戦い方](https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068) 参考

## Dataset

|Name|Detail|Ref|
|---|---|---|

## Paper

|No.|Status|Name|Detail|Date|Url|
|---|---|---|---|---|---|

## train.csv columns

|name|Explanation|
|----|----|

## TODO

|No.|Status|Name|Detail|Due Date|URL|
|---|---|---|---|---|---|
| 1 | New | 担当分の前処理 | Designation, MonthlyIncome, customer_infoのカラム処理  | 20240804 | N/A |
| 2 | New | autogluonで1サブ | とりあえずベースラインモデルとしてautogluonを投げる | 20240803 | N/A |

## Log

### 2024-08-01

- 開始
- とりあえず環境構築Done
- autogluonで1サブ。0.79ちょい。まじで前処理何もいらないのすごいよな

### 2024-08-03

- 前処理。方針は以下。
  - Designation: A-z以外の文字を持つレコードに対して、全角英字ならば半角に置き換え。その後サイドチェックしてA-z以外の文字を持つレコードのユニークな値をlistで返して、手でreplace
  - MonthlyIncome: 「数字+万」を正規表現で抽出して数値に変換。まったくマッチしないレコードについては最後にユニークな値のlistを返して、手でreplace。
  - Customer_info: 全角文字含めて一般にあり得るsep文字をすべて半角スペースに変換後、「結婚歴」「車保有」「子供の数」カラムに分割。ただこれは良い設計があまり思いつかない。スペースでwordに分けたあと、「婚」の字が入ったwordを結婚歴、「車」の文字が入ったwordを車保有カラム、残りを「残り」カラムに入れたあとまたユニーク値を見て調整かなぁ。
