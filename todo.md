[[# やること
epochとiaterationどうするか
<!-- とりあえずitereation -->
visualizer.py
    print_current_lossesは利用する．
    display current results以下のコメント以下は利用して良さそう
    
     # save images to an HTML file if they haven't been saved.
    
- [*]sare_ecnの出力層はtanhでいいのか？
res_net にDropout追加する？

- [*]dを更新するときはunitを，unitを更新する時はDのパラメータを凍結できているか？

- [] self.model_namesにunit登録．あとモデルを保存できるように書き換える．

- [] saveするネットワーク名のリストを作成する．
        必要要件
        今は一部のみしかリストに書かない→全て書くように
        そのネットワークがなかったらアサート
- []lossをディクショナリ化する
- []get current visalsで入力画像や生成画像のテンソルを辞書に渡している．ここをテンソルではなくシンボリックリンク等に変更して，画像サイズの抑制にとりくみたい．

- [ ]L159/CKandHE真っ白なトレーニングデータあり
- [] auto encoderで画像生成やったほうがいい？

