# ccgame
強化学習の演習として、CCゲームなどと呼ばれるゲームを学習させてみたものです。

## ルール
以下の4つの行動を取ることができます。
1:charge
2:guard
3:attack
4:special attack

attackの行動をするには、chargeを一回分消費し。
special attackの行動をするためには、chargeを二回分消費します。

強さは以下のように定義されます。
1<3,2<4,3<4,4<1
