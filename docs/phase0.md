# jikkenn2 Phase 0

危険部位を理解した道具の受け渡し。認識層を一切使わず、GT 姿勢だけで
「掴む → 持ち上げる → 柄を人に向けて差し出す」を最初に一周させる。
認識はその後、動いている骨格に 1 つずつ差し替える。

| | |
|---|---|
| 主張 | 危険部位の理解（刃を持ち、柄を人に向ける） |
| シミュレータ | Isaac Sim 5.1 |
| 実機 | 未定 / スコープ外 |
| 人 | 空間の 1 点として表現。メッシュは置かない |
| 配置 | 人手で並べる → 保存 → 自動で回す |
| Phase 0 の認識 | なし（Isaac の GT を使う） |

---

## 1. なぜ順序を反転するのか

前身の jikkenn1 は「認識 → 把持 → 地図 → 計画」と縦に積み上げ、
一度もロボットが動かないまま地図生成で止まった。比較対象になる baseline が
存在しないので、壊れているのがどの層かを切り分けられない状態になった。

原因は自作 ESDF だった。`observed & (distance > 0.0)` という 2 分岐で
free/blocked を決めていたが、nvblox の未更新 voxel の距離初期値は `0.0` で、
それが「障害物」側に落ちた。結果、観測領域と未観測領域の境界すべてに
1〜2 voxel の殻ができ、**机より上に約 30 万 voxel（0.30 m³）の幽霊障害物**が
浮いた。ロボットは depth からマスクで抜いてあるため、腕のシルエットの影の
境界＝殻が腕に張り付き、65 個の collision sphere のうち 50 個が常に衝突した。

決定的だったのは、**その地図の自動チェックが 6 項目すべて true で通過していた**
こと。プランナに実際に入る配列を 3D で見る手段が、リポジトリに一度も
作られなかった。点群に落として見れば 10 秒で分かった。

Phase 0 の役割はこの baseline を作ること。認識を全部外し、主張の骨格だけを
動かす。

---

## 2. シーン定義

人は机の **−y 側**に立つ。カメラはその**反対の +y 側**に置く。人が手を伸ばしても
カメラを遮らず、将来実機で人の手を見たくなったとき同じカメラで見える。
ロボットは画角の端に寄り、その遮蔽域は机の外に落ちる。

数値の正は `src/jikkenn2/scene_spec.py`。ここに書いてあるのは写しであって、
食い違ったらコードが正しい。

| 要素 | 位置 / 寸法 (m) | 備考 |
|---|---|---|
| `panda_link0` | (0.00, 0.00, 0.00) | 設置面と机上面が同じ `z = 0` |
| table | center (0.55, 0.00, −0.025) / size (0.70, 1.20, 0.05) | 手前端 `x = 0.20`、基部から 0.20 m クリア |
| camera | pos (0.50, 1.30, 1.60) → look_at (0.50, 0.00, 0.05) | 640×480 / hFOV 69° / clip 0.05–5.0 |
| handover | (0.40, −0.45, 0.35) | 基部から 0.696 m |
| `p_human` | (0.45, −1.10, 0.35) | 採点用の点。メッシュなし |
| 道具 初期 | (0.45, 0.15, 台上) | GUI で自由に動かす |
| 作業帯 | 水平距離 0.30 – 0.65 | 暫定。`reachability_map.py` が確定させる |

`scripts/validate_scene.py` が全 15 項目を機械判定する（Isaac 不要、NumPy のみ）。

```
camera half FOV        : 34.50deg horizontal, 27.27deg vertical
table corners          : |h| <= 13.43deg, |v| <= 16.36deg, depth 1.676-2.447m
robot base / top       : 13.63/0.89deg, 21.11/-25.24deg
robot fully in view    : up to z = 1.055 m  (home envelope 1.00 m)
handover distance      : 0.696 m
robot shadow on table  : 0 sampled points
```

Panda が完全に伸びると 1.19 m に達し、画角の上に出る。これは許容する ——
画角外の腕は depth に写らないので、マスクする対象にもならない。

---

## 3. 到達可能領域を先に可視化する

「適当に置くと IK が解けない」を、置く**前**に分かる状態にする。

- 机上を 2 cm グリッドでサンプル（35 x 60 = 2100 セル）
- 各セルで「上から掴む × yaw 4 通り」「横から掴む × 4 方向」= 8 姿勢を生成。
  手先は panda_hand の +z 方向に 103.4 mm 引いた位置に置く（指先が対象に閉じるように）
- cuRobo IK に投入（自己衝突 + 机との衝突込み、16800 問）
- 机の上に貼る**面ごとに色を持つメッシュ**として出力（緑 = 8 姿勢すべて解ける / 黄 = 一部 / 赤 = 不可）

以後、配置の議論は「この座標どうですか」ではなく「緑の中です」で済む。

世界モデルに入れるのは**机だけ**。障害物は配置ごとに動くので、地図に焼き込むと
何かをドラッグした瞬間に嘘になる。これは配置の補助であって安全ゲートではない
（実シーンの衝突判定はプランナがやる）。

`--dry-run` を付けると cuRobo を一切呼ばず灰色のオーバーレイだけ出す。
色を信じる前に「オーバーレイがビューポートに出るか」だけを切り分けるために使う。

---

## 4. 配置は人手、実行は自動

誰が何の正なのかを固定する。

| ファイル | 正 | 誰が変える |
|---|---|---|
| `src/jikkenn2/scene_spec.py` | レイアウトの正（机・カメラ・handover・道具の定義） | コード変更 |
| `assets/scene.usd` | `scene_spec.py` からの**生成物**。git には入れない | `build_scene_usd.py` |
| `arrangements/*.json` | 物体配置の正。**git に入れる** | GUI で人が並べる |

GUI での操作は道具と障害物を動かすことだけで、**ステージ自体は保存しない**
（`Ctrl+S` は押さない）。机やカメラを変えたくなったら `scene_spec.py` を直して
`build_scene_usd.py` を回し直す。こうすれば手作業の配置と構造の定義が混ざらない。

```
assets/scene.usd                          ← scene_spec から生成。GUI で人が動かす
        │  置いて、ターミナルで Enter
        ▼
arrangements/arr_001.json … arr_020.json  ← 全物体の pose（リポジトリに入れる）
        │
        ▼
for arr in arrangements: 復元 → capture → plan → execute → score   ← 完全自動
```

配置＝人手（20 通りで 20 分）、実行＝完全自動。これで「20 配置 × N 道具」の
成功率が主張を支える形で出る。

**物理を止めた状態で並べ、数フレーム step して落ち着かせてから pose を記録する。**
手で置くと必ず浮くか刺さるので、こうすれば物理的に成立した配置しか保存されない。

### Isaac Sim GUI で覚えること

1. `python scripts/arrange_scene.py --scene assets/scene.usd` で GUI 起動、ステージを開いた状態で待機
2. 視点 — 右ドラッグ = 回転、中ドラッグ = 平行移動、ホイール = ズーム、選択して `F` で寄る
3. 選択 — Stage ツリーで `/World/Tools/proxy_tool` を選ぶか、ビューポートで直接クリック
4. 移動 — `W` = 移動ギズモ、`E` = 回転ギズモ。数値入力は Property パネルの Transform。**Play は押さない**
5. 緑の中に置く（3 節のオーバーレイ）。赤の上に置いた配置は保存時に警告
6. ターミナルに戻って `Enter`。現在の pose が `arrangements/arr_NNN.json` に書き出される。
   `l` で保存せずに一覧、`q` で終了。**1 回の GUI セッションで何個でも保存できる**

保存時に各オブジェクトの判定が 1 行ずつ出る。

```
  proxy_tool     (+0.450, +0.150, +0.023)  OK
  obstacle_a     (+0.450, -0.180, +0.050)  OK
  obstacle_b     (+1.600, +0.280, +0.060)  WARN: off table, outside reach band (1.60 m)
SAVED /home/suzutaro/projects/jikkenn2/arrangements/arr_001.json  [1 this session]
```

`WARN` が付いた配置も保存はされる（意図的に届かない配置を作りたい場合があるため）。
記録されるのは**人が置いた姿勢そのもの**で、物理の安定化は行わない。落ち着かせるのは
`capture_scene.py` の役目で、そちらが安定後の姿勢を試行データと一緒に記録し、ズレが
大きければ警告する。こうすると対話ツール側で物理を回さずに済み、GUI が固まらない。

---

## 5. 採点仕様

`掴めた + 運べた` だけでは主張を支えない（それはただの pick and place）。
部位を採点に入れる。`p_human` が点なので、人形を置かずに全項目 GT から自動採点できる。

| # | 条件 | 判定 |
|---|---|---|
| 1 | 意図した部位を掴んだ | 把持中心が危険部位 AABB 内（GT） |
| 2 | 落としていない | 持ち上げ後 2 s、物体が gripper に追従 |
| 3 | 衝突していない | 軌道中、ロボット ↔ 机 / 障害物の接触ゼロ |
| 4 | **柄が人を向いている** | 柄軸と `p_human` 方向のなす角 < 30° |
| 5 | **刃が人を向いていない** | 刃軸と `p_human` 方向のなす角 > 90° |
| 6 | handover 姿勢に到達 | 位置誤差 < 2 cm、姿勢誤差 < 10° |

### 道具

Phase 0 の道具は **proxy tool（柄の箱 + 頭の箱）**。実メッシュは使わない。

- 部位が定義上ぴったり決まるので採点 1・4・5 に曖昧さが入らない
- アセット依存ゼロで今日から動く
- Phase 2 で実メッシュに差し替えたときの baseline になる

頭 60×45×45 mm（Panda の開き幅 80 mm に収まる）、柄 140×28×28 mm。
ロボットは**頭（危険部位）を掴み、柄（安全部位）を人に向ける**。

部位アノテーション `assets/tools/proxy_tool.json` は `scene_spec.py` から生成される
**採点専用の GT** で、認識側は絶対に参照しない。

実メッシュは Phase 2 から。順番は **power_drill → scissors → knife**。
薄物から始めると、薄物の depth・薄物の部位分割・薄物の把持が同時に来る。

---

## 6. スクリプト構成

環境は 2 つ。8 GB VRAM では Isaac Sim・SAM3・GraspGenX・cuRobo の同時常駐は
物理的に無理なので、**段階ごとに別プロセス・中間ファイル受け渡し**は維持する
（jikkenn1 のこの判断は正しかった）。減らすのは torch が 3 バージョンに割れている
環境の分裂のほうで、Backend A / B を落とせば 4 → 2 になる。

- **A** = `env_isaaclab`（Isaac Sim、のちに SAM3）
- **B** = GraspGenX venv（cuRobo、のちに GraspGenX）

| スクリプト | 環境 | 役割 | 状態 |
|---|---|---|---|
| `validate_scene.py` | — | 画角・レイアウト・クリアランスの機械判定 | 済 |
| `build_scene_usd.py` | A | `assets/scene.usd` を生成 | 済（USD 部分は検証済み） |
| `reachability_map.py` | B | 到達可能領域の計算とオーバーレイ | 済（cuRobo 呼び出しは未検証） |
| `arrange_scene.py` | A | GUI 起動 → 人が配置 → arrangement 保存 | 済（ロジックは検証済み） |
| `capture_scene.py` | A | 配置復元 → RGB-D・関節・GT 姿勢の保存 | 未 |
| `plan_handover.py` | B | GT 姿勢 → 把持姿勢 → cuRobo ESDF → 軌道 | 未 |
| `execute_handover.py` | A | 軌道実行・把持・持ち上げ・handover・動画 | 未 |
| `score_trial.py` | — | 6 項目の自動採点 | 未 |
| `run_phase0.py` | — | arrangements をループ | 未 |

### コマンド列

```bash
# 一度だけ ─ 検証して、シーンを作り、到達域を描く
python scripts/validate_scene.py

conda activate env_isaaclab
python scripts/build_scene_usd.py --output assets/scene.usd

source ~/GraspGenX/.venv/bin/activate
python scripts/reachability_map.py --scene assets/scene.usd \
       --output outputs/reachability --overlay assets/overlay_reach.usd

# 配置する（人手） ─ 置いて Enter、を 20 回
conda activate env_isaaclab
python scripts/arrange_scene.py --scene assets/scene.usd --output arrangements/

# 1 試行
python scripts/capture_scene.py --scene assets/scene.usd \
       --arrangement arrangements/arr_001.json --output outputs/trial_001
source ~/GraspGenX/.venv/bin/activate
python scripts/plan_handover.py --capture outputs/trial_001 \
       --tool assets/tools/proxy_tool.json --output outputs/trial_001/plan
conda activate env_isaaclab
python scripts/execute_handover.py --capture outputs/trial_001 \
       --plan outputs/trial_001/plan --record-video
python scripts/score_trial.py --trial outputs/trial_001

# 全部回す
python scripts/run_phase0.py --arrangements arrangements/ --output outputs/phase0
```

### Franka の USD が見つからないとき

Isaac はリリースごとにマニピュレータのアセット位置を変える。`build_scene_usd.py` は
既知のパスを順に試し、全部外れたら `/Isaac/Robots` を深さ 3 まで探索する。それでも
駄目なら手で探す。

```bash
python scripts/build_scene_usd.py --list-assets                    # /Isaac/Robots
python scripts/build_scene_usd.py --list-assets /Isaac/Robots/Franka
python scripts/build_scene_usd.py --franka-usd "omniverse://.../franka.usd"
```

どこで落ちても `outputs/scene_build.json` に status・例外・traceback が残る。
**何も残さずに落ちるスクリプトは書かない。**

テストは Isaac も GPU も要らない（`usd-core` があれば USD の検証まで走る）。

```bash
python -m pytest -q
```

---

## 7. jikkenn1 から移すもの・捨てるもの

| 対象 | 判定 | 理由 |
|---|---|---|
| `geometry.py` | 移植済 | 往復誤差 1.2e-4 px / 2.6e-7 m で検証済み |
| `capture.py` / `robot_state.py` | 移植 | Isaac 公式 API 経由の RGB-D + 関節状態保存 |
| `segmentation.py` | 移植（Phase 2） | SAM3 の使い方は素直。マスクのはみ出し対策を追加 |
| `grasp_candidates.py` | 移植（Phase 2） | GraspGenX 連携と frame 変換が検証済み |
| `scene_layout.py` | 発想のみ | 集約と自動チェックの発想は良い。座標は USD に移した |
| `curobo_bridge.py` | 移植 | 関節名によるマッピング |
| `curobo_map_capture.py` | 一部 | RobotSegmenter のみ。「対象物を depth から消す」は attach に置換 |
| `conservative_esdf.py` | 破棄 | 自作 ESDF。幽霊障害物の発生源 |
| Backend A / B 一式 | 破棄 | cuRobo `Mapper.compute_esdf()` が同じことを既にやっている |
| cuRobo #699 パッチ | 再確認 | 新しい cuRobo で修正済みなら不要 |
| safety JSON 契約 | 縮小 | 何も実行していない段階では守るものがない |

---

## 8. 作業原則

1–7 は jikkenn1 から継承。実際によく機能していた。

1. 問題が起きたら、まず公式実装・GitHub・Issue・論文・類似プロジェクトを探す
2. 自前実装が必要だと思ったときも、本当に既存実装がないか再確認する
3. 複数の既存手法があるなら比較してから選ぶ
4. 動かないからという理由だけで、閾値や座標を恣意的に変更しない
5. 修正が次の問題を生まないか考える
6. 本当の分岐や壁に到達したら、勝手に決めず確認する
7. 衝突や座標系を曖昧にしない

ここから先は jikkenn1 の失敗から追加したもの。

8. **自前実装したら、それを 3D で目視できる出力を同時に作る。** JSON の自己整合チェックは検証ではない
9. **GT は認識入力に使わない。GT は検証に使う。** 両者をコード上で明示的に分ける
10. **各 Phase は直前の Phase を baseline として数値比較する。** 差分が出たら、差し替えた層が原因
11. **空間配置は人が決める。** AI は到達可能性のように機械判定できる形でしか提案しない
12. **実行していない段階の安全ゲートは最小限にする。** その労力は幾何テストに回す
13. **公式に無い機能が欲しくなったら、それは「自作の合図」ではなく「その要求は今本当に必要か」を疑う合図。**
    jikkenn1 で原則 1・2 が唯一破られたのがここだった。「保守的な地図（unknown=blocked）が
    欲しい」という要求が先にあり、公式に無かったので自作に進んだ。だが単視点 + conservative は
    原理的に詰む —— 机の裏・物体の影・ロボットの後ろは永久に unknown なので全部壁になる。
    要求のほうが間違っていた

---

## 9. Phase 0 の完了条件

- [ ] 1 配置で採点 6 項目すべて pass、実行動画がある
- [ ] 20 配置が保存済みで、`run_phase0.py` が無人で完走し成功率が出る
- [ ] `validate_scene.py` が pass し、到達域オーバーレイが成果物として残っている
- [ ] SAM3・GraspGenX・nvblox を一度も呼んでいない（呼んでいたら Phase 0 ではない）
- [ ] この文書が最新（README には書かない。公開リポジトリなので README は警告文のみ）

### この先

| Phase | 差し替える層 | 比較対象 |
|---|---|---|
| 1 | 地図：GT ボックス → cuRobo Mapper の実測 ESDF | Phase 0 の軌道・成功率 |
| 2 | 物体位置：GT → SAM3 + GraspGenX | Phase 1 の姿勢誤差 |
| 3 | 部位分割：アノテーション → SAM3 の blade / handle | Phase 2 の部位正解率 |
| 4 | 入口：VLM、道具 5 種 | Phase 3 |
| 5 | 多視点 → RealSense → 実機 | Phase 4 |

**Phase 1 の受け入れテスト（必須）**：capture 時の関節姿勢で 65 個の collision sphere が
**全て衝突ゼロ**。sim では腕が自由空間にあることが既知なので、1 個でも当たったら地図が
壊れている。加えて、プランナに渡す ESDF そのものを Viser に描き、ロボットと重ねて目視する。
この 1 行のテストがあれば、jikkenn1 の問題は初日に落ちていた。
