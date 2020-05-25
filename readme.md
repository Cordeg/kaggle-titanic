
## 手法
決定木を利用。
['Name', 'Ticket', 'Cabin']の情報は無視した。

## 今後の改善点
### 値を持たないサンプルをどう扱うか。
1. 平均値を入れる。
2. すべてのノードに含まれる（すべてのデータを持つ）ようにして、最後に分岐した分を希釈する。
3. ゴミデータとしてそのサンプルを無視する。

## 参考にしたサイト：
https://pythondatascience.plavox.info/scikit-learn/scikit-learn%E3%81%A7%E6%B1%BA%E5%AE%9A%E6%9C%A8%E5%88%86%E6%9E%90

ipynbファイルをpyの変換 command
> jupyter nbconvert --to script *.ipynb
