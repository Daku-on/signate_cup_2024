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
| 1 | Done | 担当分の前処理 | Designation, MonthlyIncome, customer_infoのカラム処理  | 20240804 | N/A |
| 2 | Done | autogluonで1サブ | とりあえずベースラインモデルとしてautogluonを投げる | 20240803 | N/A |
| 3 | New | NN model作成 | とりあえずNNのベースラインモデルとしてKaggleのモデルまんまを流用する | 20240812 | https://www.kaggle.com/code/takakookuda/neural-network-based-solution-tensorflow-98-5 |

## Log

### 2024-08-01

- 開始
- とりあえず環境構築Done
- autogluonで1サブ。0.79ちょい。まじで前処理何もいらないのすごいよな

### 2024-08-03

- 前処理。方針は以下。
  - Designation: NULLなし。A-z以外の文字を持つレコードに対して、全角英字ならば半角に置き換え。その後サイドチェックしてA-z以外の文字を持つレコードのユニークな値をlistで返して、手でreplace
  - MonthlyIncome: NULLあり。ただこれは無回答の場合もあるだろうし、56しかないのでNNに回すときはDesignationクラスタの平均で埋めるとかするべきだな。0ではない。「数字+万」を正規表現で抽出して数値に変換。まったくマッチしないレコードについては最後にユニークな値のlistを返して、手でreplace。
  - Customer_info: NULLなし。全角文字含めて一般にあり得るsep文字をすべて半角スペースに変換後、「結婚歴」「車保有」「子供の数」カラムに分割。ただこれは良い設計があまり思いつかない。スペースでwordに分けたあと、「婚」の字が入ったwordを結婚歴、「車」の文字が入ったwordを車保有カラム、残りを「残り」カラムに入れたあとまたユニーク値を見て調整かなぁ。

### 2024-08-06

- 前処理関数がバグってたので直した。"nan"文字列でmatching判定していたので常にエラーを吐く状態になってた。
- word2vec + SVDで次元削減、を書いた。いいえ。以下を参考にChatGPTに書いてもらいました。https://www.kaggle.com/code/abdmental01/bank-churn-lightgbm-and-catboost-0-8945

### 2024-08-08

- 前処理関数をまとめた。
  - TODO: polarsにしてスキーマ決めて流し込もう。

### 2024-08-11

- とりあえずtoy NN modelをまわすところまで行った。(nb02)
  - 結果は0.8。trainが下がってるのにvalが上がり始めているので、普通にモデルの表現力足りてない説は若干ある
  - word2vecいれるの忘れてた
  - GCondNetというのがsmall tableに効果的らしい。GPTに書いてもらったので論文読みつつできれば使ってみたいですねぇ

### 2024-08-12

- 意外とテーブルデータ特化のNNが色々あるみたいなので試す。
  - 例えば[テーブルデータ用ニューラルネットワークは勾配ブースティング木にどこまで迫れるのか？（本編）](https://note.com/japan_d2/n/nf8023bf90f8f)
  - まずはtabnet

### 2024-08-15

- 提案すべきProductPitchedをうまく予測できればいい特徴量になるんでは？というyanaboからのアイデアを強化学習でやってみた
  - basicを提案するだけの意味ないエージェントができてしまった。報酬をいじってもだめっぽい。
