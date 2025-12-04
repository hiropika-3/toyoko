import os
import time
import json
import math
import datetime as dt
import tempfile
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import random
import requests
import yaml
import ast

from src.speech_analyzer import SpeechAnalyzer

# 詳細設定エリアをアプリケーション側で ON/OFF
#  - DETAIL_PANEL_VISIBLE=1 で表示
#  - 未設定 or 0 なら非表示（デフォルト）
DETAIL_PANEL_VISIBLE = os.getenv("DETAIL_PANEL_VISIBLE", "0") == "1"

# 音声特徴のうち「レーダーチャートで扱う5項目だけ」を抽出するための許可リスト
ALLOWED_KEYS = {"速さ", "抑揚", "音量", "明瞭さ", "間"}

# トヨコの追加フィードバック（ファイルがなければ非表示）
TEMPLATES_PATH = os.getenv("ADVICE_TEMPLATES_PATH", "src/advice_templates.yaml")
_TPL_CACHE = {"mtime": None, "data": None}

# 安全な条件式評価（比較/論理/数値/識別子/括弧/四則のみ許可）
_ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
}

# 今日のあなたにVoicyからに使うURLの一覧
VOICY_YAML_PATH = os.getenv("VOICY_EPISODES_PATH", "src/voicy_episodes.yaml")
_VOICY_CACHE = {"mtime": None, "episodes": []}

# （任意）LLM API 呼び出し（未設定なら無効）
MYGPT_API_BASE = os.getenv("MYGPT_API_BASE", "").rstrip("/")
MYGPT_API_KEY = os.getenv("MYGPT_API_KEY", "")
MYGPT_MODEL_ID = os.getenv("MYGPT_MODEL_ID", "")

CUSTOM_CSS = """
    #rec-wrapper {
      border: 3px solid #00bcd4;              /* 強めの青緑の枠 */
      border-radius: 18px;
      padding: 18px;
      background: linear-gradient(135deg, #e0f7fa, #f1fcff);
      box-shadow: 0 0 16px rgba(0, 188, 212, 0.7);
      animation: recGlow 1.4s ease-in-out infinite alternate;  /* ふわっと光る */
    }
    @keyframes recGlow {
      0%   { box-shadow: 0 0 10px rgba(0,188,212,0.4); }
      100% { box-shadow: 0 0 28px rgba(0,188,212,0.9); }
    }

    /* ヒーローセクション */
    .hero {
      display: flex;
      align-items: center;
      gap: 2rem;
      padding: 2.2rem 2rem;
      border-radius: 24px;
      background: linear-gradient(135deg, #FFE8D6, #FFF6EF);
      box-shadow: 0 4px 16px rgba(0,0,0,0.06);
      margin-bottom: 1.2rem;
    }

    .hero img {
      width: 180px;
      height: auto;
      border-radius: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .hero-text h1 {
      font-size: 1.8rem;
      margin: 0 0 0.6rem;
      color: #C95A2A;
    }

    .hero-text p {
      font-size: 1.05rem;
      line-height: 1.55;
      color: #6B4F4F;
    }

    .feature-cards {
      display: flex;
      gap: 1.2rem;
      flex-wrap: wrap;
      margin-bottom: 1.8rem;
    }

    .feature-card {
      flex: 1 1 calc(33% - 1.2rem);
      min-width: 220px;
      background: #FFF7F1;
      border-radius: 18px;
      padding: 1.2rem 1.4rem;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .feature-card h3 {
      margin: 0 0 0.4rem;
      color: #C95A2A;
    }

    .feature-card p {
      color: #6B4F4F;
      font-size: 0.95rem;
      line-height: 1.5;
    }
 
    #hero-image {
      max-width: 100%;
    }

    /* 音量スライダー（range入力）を非表示 */
    #audio-input input[type="range"] {
      display: none !important;
    }

    /* ===== メニュー共通スタイル ===== */
    .top-menu, .mobile-menu {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* PC メニュー（横並び） */
    .top-menu {
        display: flex;
        gap: 24px;
        padding: 14px 22px;
        background: linear-gradient(135deg, #f7e9d7 0%, #eed9c4 45%, #ffffff 100%);
        border-bottom: 1px solid #e0d2c2;
    }

    /* 共通リンクスタイル */
    .top-menu a, .mobile-menu a {
        text-decoration: none;
        color: #6a4d32;
        font-weight: 600;
        font-size: 15px;
        padding: 4px 6px;
        border-radius: 6px;
    }

    .top-menu a:hover,
    .mobile-menu a:hover {
        background: rgba(255, 255, 255, 0.45);
    }

    /* ハンバーガーとトグル用チェックボックス */
    .menu-toggle {
        display: none;  /* チェックボックス自体は隠す */
    }

    .hamburger {
        display: none;  /* デフォルトはPCでは非表示 */
        font-size: 30px;
        cursor: pointer;
        padding: 12px 20px;
        background: linear-gradient(135deg, #f7e9d7 0%, #eed9c4 45%, #ffffff 100%);
        color: #6a4d32;
        border-bottom: 1px solid #e0d2c2;
        user-select: none;
    }

    /* スマホ用メニュー（デフォルトは非表示） */
    .mobile-menu {
        display: none;
        flex-direction: column;
        gap: 16px;
        padding: 16px 22px;
        background: linear-gradient(135deg, #f7e9d7 0%, #eed9c4 45%, #ffffff 100%);
        border-bottom: 1px solid #e0d2c2;
    }

    /* --- 768px 以下をスマホ表示とする --- */
    @media (max-width: 768px) {
        .top-menu {
            display: none;         /* PC メニューは隠す */
        }
        .hamburger {
            display: block;        /* 三本線を表示 */
        }
        /* チェックが入ったらモバイルメニューを開く */
        .menu-toggle:checked ~ .mobile-menu {
            display: flex;
        }
    }

    /* 768px より大きいときは、モバイルメニューは常に非表示 */
    @media (min-width: 769px) {
        .mobile-menu {
            display: none !important;
        }
    }

    #banner-img {
        width: 100%;
        max-width: 300px;   /* PC の最大幅 */
        display: block;
        margin: 0 auto;     /* 中央寄せ */
        border-radius: 12px;
    }

    .custom-video video {
        width: 100% !important;
        max-width: 500px !important;  /* PCでの最大幅 */
        border-radius: 12px;
        display: block;
        margin: 0 auto;
    }
"""

CUSTOM_JS = """
function toggleMenu() {
    const menu = document.getElementById("mobile-menu");
    if (!menu) return;

    if (menu.style.display === "flex") {
        menu.style.display = "none";
    } else {
        menu.style.display = "flex";
    }
}
"""

# ─────────────────────────────────────────────────────
# Voicy episodes YAML 読み込み（ホットリロード対応）
# ─────────────────────────────────────────────────────
def load_voicy_episodes(path=VOICY_YAML_PATH):
    """YAML が更新されたら自動で再読み込みする"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []

    # 更新がなければキャッシュを返す
    if _VOICY_CACHE["mtime"] == mtime:
        return _VOICY_CACHE["episodes"]

    # 読み直し
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            episodes = data.get("episodes", [])
    except Exception:
        episodes = []

    _VOICY_CACHE["mtime"] = mtime
    _VOICY_CACHE["episodes"] = episodes
    return episodes

def _pick_target_key_from_features(features: dict) -> str:
    """
    features から「今伸ばしたいポイント」を決める。
    速さ / 抑揚 / 音量 / 明瞭さ / 間 のような 0〜1 の数値を想定。
    最も低いスコアの項目を「伸ばしたい」とみなす。
    """
    if not isinstance(features, dict):
        return ""

    # 数値だけ抽出（list や None が混ざっていても安全化）
    numeric_feats = {
        k: float(v)
        for k, v in features.items()
        if isinstance(v, (int, float))
    }

    if not numeric_feats:
        return ""

    # スコアが最も低い項目を選ぶ
    weakest_name = min(numeric_feats, key=numeric_feats.get)
    return weakest_name  # 例: "抑揚"

def _build_voicy_intro_text(target_key: str) -> str:
    """
    「どのポイントを伸ばしたいか」に応じて、
    Voicyセクションの冒頭に入れる一言を切り替える。
    """
    mapping = {
        "抑揚": "今日は **「抑揚」や「メリハリ」を伸ばしたいあなたに** ぴったりの放送を選びました。",
        "速さ": "今日は **「話す速さ」や「テンポ」を整えたいあなたに** 合う放送です。",
        "音量": "今日は **「声の大きさ・エネルギー感」を高めたいあなたに** 聴いてほしい１本です。",
        "明瞭さ": "今日は **「言葉の聞き取りやすさ」や「伝わり方」を磨きたいあなたに** 合う放送です。",
        "間": "今日は **「間の取り方」や「リズム感」を良くしたいあなたに** 合う放送です。",
    }

    # ターゲットが未特定なら汎用メッセージを返す
    return mapping.get(
        target_key,
        "今日は **今のあなたの声に寄り添う１本** を選びました。"
    )

def build_voicy_section(features: dict, dbfs: float) -> str:
    """
    特徴量と dBFS に応じて、YAML から Voicy を選んで紹介する
    """
    episodes = load_voicy_episodes()
    if not episodes:
        return ""  # YAML が空なら何も出さない

    target_key = _pick_target_key_from_features(features)

    # 伸ばしたいポイントに合うものを優先
    candidates = []
    if target_key:
        for ep in episodes:
            if target_key in ep.get("targets", []):
                candidates.append(ep)

    # 音量がかなり小さい人は別の候補も追加
    if not candidates and dbfs < -30:
        for ep in episodes:
            if "音量" in ep.get("targets", []) or "自信" in ep.get("targets", []):
                candidates.append(ep)

    # それでも無ければ全体から
    if not candidates:
        candidates = episodes

    ep = random.choice(candidates)

    intro = _build_voicy_intro_text(target_key)

    md = f"""
---

## 今日のあなたに 🎧 Voicy から

{intro}

▶️ **[{ep['title']}]({ep['url']})**
阪急電車の声の人・下間都代子（Voicy）

**おすすめ理由：**
- {ep['reason']}
- 解析したあなたの声の状態と相性が良い１本です。
"""
    return md

def call_mygpt(system_prompt: str, user_prompt: str, timeout: float = 8.0) -> str:
    """任意の LLM API 呼び出し。未設定なら空文字を返す。"""
    if not (MYGPT_API_BASE and MYGPT_API_KEY and MYGPT_MODEL_ID):
        return ""
    url = f"{MYGPT_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {MYGPT_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MYGPT_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 600,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def build_llm_prompts(metrics: dict) -> tuple[str, str]:
    system = (
        "あなたはフリーアナウンサーで声の総合プロデューサー、"
        "そして阪急電車の声の人・下間都代子です。"
        "ラジオのパーソナリティのように、明るくテンション高めで話します。"
        "相手は長年応援しているリスナーさん。"
        "フランクだけど丁寧、ちょっと関西ノリで、前向きなエネルギーを乗せてください。"
        "助言は3〜6項目程度、箇条書きで具体的に。"
        "・ダメ出しではなく『こうするともっと素敵になるよ』という言い方にする\n"
        "・専門用語は避けて、初心者にもわかる言葉で\n"
        "・ところどころ「〜だよ」「〜してみてね」「〜してあげて」など、"
        "やわらかい語尾を混ぜる\n"
    )

    user = (
        "次の音声解析の客観指標を踏まえて、"
        "都代子さんとして追加フィードバックを作成してください。\n"
        "【トーンの指定】\n"
        "・仲の良いリスナーさんに話しかける感じ\n"
        "・上から目線ではなく、隣で一緒に練習しているコーチのイメージ\n"
        "・1文をあまり長くしすぎない\n"
        "【内容の条件】\n"
        "・短く要点的に（3〜6項目）\n"
        "・語尾は助言調で優しく「〜してみて」「〜してあげてね」など\n"
        "・具体的行動（例：キーワードの前で0.3秒だけ間を置いてみて など）\n\n"
        "・最後に全体的な感想とポジティブなフィードバックを都代子節で長めの文章で面白おかしく補足してください\n\n"
        f"{json.dumps(metrics, ensure_ascii=False, indent=2)}"
    )
    return system, user

# ─────────────────────────────────────────────────────
# YAML テンプレート: ロード & 安全評価 & レンダリング
# ─────────────────────────────────────────────────────

def load_templates_if_changed(path=TEMPLATES_PATH):
    """ファイル更新を検知して再ロード（ホットリロード）"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _TPL_CACHE["mtime"] = None
        _TPL_CACHE["data"] = None
        return None
    if _TPL_CACHE["mtime"] != mtime:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _TPL_CACHE["mtime"] = mtime
            _TPL_CACHE["data"] = data
        except Exception:
            _TPL_CACHE["mtime"] = None
            _TPL_CACHE["data"] = None
    return _TPL_CACHE["data"]

def _safe_eval_expr(expr: str, env: dict) -> bool:
    """astで構文木を検査し、許可ノードのみ評価"""
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return False

    def _check(node):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"disallowed node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            _check(child)

    try:
        _check(tree)
        return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env))
    except Exception:
        return False


def render_rule_based_feedback(metrics: dict) -> str:
    """
    YAMLの sections[].rules[].if を安全に評価し、該当テキストをMarkdownで返す。
    1つのセクションにつき、ヒットしたルールを箇条書きで並べる。
    ヒットが総ゼロなら空文字を返す。
    """
    data = load_templates_if_changed()
    if not data or "sections" not in data:
        return ""

    env = dict(metrics)  # dbfs, clip_ratio, など

    parts = []
    for sec in data.get("sections", []):
        heading = sec.get("heading", "トヨコの飛び蹴りフィードバック（テンプレート）")
        lines = []
        for rule in sec.get("rules", []):
            cond = str(rule.get("if", "")).strip()
            text = rule.get("text", "")
            if not cond or not text:
                continue

            if _safe_eval_expr(cond, env):
                # text がリストならランダムで1件選ぶ
                if isinstance(text, list):
                    selected_text = random.choice(text)
                else:
                    selected_text = str(text)
                lines.append(f"- {selected_text}")
        if lines:
            parts.append(f"\n---\n\n## {heading}\n" + "\n".join(lines))

    return "".join(parts)

def _describe_level(v: float) -> str:
    """0〜1の値を日本語のざっくりした評価に変換"""
    if v >= 0.85:
        return "かなり強め"
    elif v >= 0.65:
        return "やや強め"
    elif v >= 0.45:
        return "ちょうど良い"
    elif v >= 0.25:
        return "やや控えめ"
    else:
        return "かなり控えめ"

def build_graph_comments(
    base_result: dict,
    peak: float,
    rms: float,
    dbfs: float,
    clip_ratio: float,
    silence_ratio: float,
    crest_factor: float,
    x_for_spec: np.ndarray,
    sr: int,
) -> tuple[str, str, str]:
    """
    波形 / レーダー / スペクトログラムごとに
    「読み方」と今回の声のコメントを都代子さん風に返す。
    戻り値: (wave_md, radar_md, spec_md)
    """

    # --- 安全にスカラー化（list や ndarray でも平均値を取る） ---
    def _scalar(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0:
                return 0.0
            return float(np.mean(v))
        try:
            return float(v)
        except Exception:
            return 0.0

    #features = base_result.get("features", {}) or {}
    # features からレーダーチャート対象だけを抜き出す
    features = {
        k: v
        for k, v in (base_result.get("features", {}) or {}).items()
        if k in ALLOWED_KEYS
    }
    feats_scalar = {k: _scalar(v) for k, v in features.items()}

    # 強い項目・弱い項目をざっくり把握
    sorted_feats = sorted(feats_scalar.items(), key=lambda kv: kv[1], reverse=True)
    strongest = sorted_feats[0] if sorted_feats else None
    weakest = sorted_feats[-1] if len(sorted_feats) > 1 else None

    # 発話時間（ざっくり秒）
    if x_for_spec is not None and sr > 0 and getattr(x_for_spec, "size", 0) > 0:
        duration_sec = len(x_for_spec) / sr
    else:
        duration_sec = 0.0

    # ── 音量まわり（波形） ─────────────────
    if dbfs > -12:
        loud_comment = (
            "かなりしっかり響いていますね。"
            "オンライン会議だと、マイク入力をひと目盛りだけ下げても十分届きそうです。"
        )
    elif -25 <= dbfs <= -15:
        loud_comment = (
            "とっても聞き取りやすい、ちょうどいい音量感です。"
            "この感じを“自分の標準”として、身体に覚えさせてあげましょう。"
        )
    else:
        loud_comment = (
            "やや控えめな音量です。"
            "大事なキーワードの前だけ、すこ〜しだけ声を前に押し出すイメージで話してみてくださいね。"
        )

    if clip_ratio > 0.02:
        clip_comment = (
            "ところどころ音が割れかけています。"
            "ワッと感情が高ぶったときこそ、ひと呼吸おいてから話し始めるクセをつけてあげましょう。"
        )
    else:
        clip_comment = (
            "音割れはほとんどなく、耳に心地よい音量バランスです。"
            "この安定感は、とっても大きな強みですよ。"
        )

    if silence_ratio > 0.65:
        silence_comment = (
            "間がたっぷりめのスタイルです。"
            "説明シーンでは、今より半歩だけテンポを上げてあげると、ぐっと聞きやすくなります。"
        )
    elif silence_ratio < 0.3:
        silence_comment = (
            "間が少なめで、情報がぎゅっと詰まった話し方になっています。"
            "キーワードの前で 0.3 秒だけフッと止まる“間のごほうび”を入れてみましょう。"
        )
    else:
        silence_comment = (
            "間の取り方が自然で、安心して聞いていられるテンポ感です。"
            "このリズムは、そのまま大切に育てていきたいですね。"
        )

    wave_md = f"""
### 波形グラフの読み方と、今回のボリューム

横軸が **時間**、縦軸が **声の大きさ** です。  
山がぐっと高くなっているところは、気持ちが前に出ているところ。  
少し平らなところは、息を吸ったり、間を置いている部分と考えてくださいね。

**トヨコのひとこと📝**  
- 音量について：{loud_comment}  
- 音割れについて：{clip_comment}  
- 間の取り方について：{silence_comment}
""".strip()

    # ── レーダーチャート ─────────────────
    radar_lines = []
    if strongest:
        radar_lines.append(
            f"- **「{strongest[0]}」が今回のいちばんの伸びしろ（というか、すでに“武器”）**として出ています。"
            " ここは遠慮せず、どんどん出していきましょう。"
        )
    if weakest:
        radar_lines.append(
            f"- 反対に **「{weakest[0]}」は少し控えめ**。"
            " ここを“いきなり完璧”ではなく、まずは 10% だけ意識してみる…くらいがちょうどいいです。"
        )
    if not radar_lines:
        radar_lines.append(
            "- 全体的にバランス型の声になっています。"
            " ここからは「どこをもっと目立たせたいか？」を決めて、少しずつメリハリをつけていきましょう。"
        )

    radar_md = (
        "### レーダーチャートの見方と、今回の強み\n\n"
        "「速さ」「抑揚」「音量」「明瞭さ」「間」など、"
        "声の要素をまとめて見られるのがレーダーチャートです。\n\n"
        "外側に張り出しているほど、その項目が“よく出ている”イメージで見てくださいね。\n\n"
        "今回のあなたの傾向は…\n\n"
        + "\n".join(radar_lines)
    )

    # ── スペクトログラム ─────────────────
    if duration_sec <= 0:
        spec_comment = (
            "今回の録音時間はごく短めでした。"
            "30 秒〜1 分ほど話してもらえると、声のクセや高さの変化がもっとはっきり見えてきます。"
        )
    elif duration_sec < 20:
        spec_comment = (
            f"今回の録音時間は約 **{duration_sec:.1f} 秒**。"
            " ウォーミングアップにはちょうどいい長さですね。"
            " 次は少し長めの一息トークにも挑戦してみましょう。"
        )
    else:
        spec_comment = (
            f"今回の録音時間は約 **{duration_sec:.1f} 秒**。"
            " じっくり話してくださったので、声の特徴や安定感がしっかり表れています。"
        )

    spec_md = f"""
### 声の高さ・響きの見方（スペクトログラム）

縦軸が **周波数（声の高さ）**、横軸が **時間**、  
色の濃さが **エネルギーの強さ** です。

- 低いところがしっかり色づいていると「落ち着いた声」の土台ができています。
- 高いところにも色が出ていると、「明るさ・華やかさ」がプラスされます。

{spec_comment}

**トヨコのひとこと🎧**  
今日はこのグラフを眺めながら、  
「もう少し明るくしたい日は、最初の一声だけちょっと高めで入ってみようかな」  
そんなふうに、“声のスイッチ”を決めてあげると、ぐっとコントロールしやすくなりますよ。
""".strip()

    return wave_md, radar_md, spec_md

# ─────────────────────────────────────────────────────
# ユーティリティ（正規化/保存/描画）
# ─────────────────────────────────────────────────────
def safe_peak(x: np.ndarray) -> float:
    """NaN/Infを無視して安全にピークを求める"""
    if x.size == 0:
        return 0.0
    xp = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(np.abs(xp)))


def normalize_for_saving(x: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    """
    保存用の安全正規化（解析用の元信号は変更しない）
    - 入力: float32/float64 [-1,?] のモノラルまたはステレオ (N,) or (N,2)
    - 出力: float32 [-1,1] 相当（target_peak までスケール）
    """
    if x is None or x.size == 0:
        return np.zeros(1, dtype=np.float32)
    xp = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = safe_peak(xp)
    if peak < 1e-6:
        return xp
    if peak > 1.0:
        xp = xp / peak
        peak = 1.0
    scale = target_peak / max(peak, 1e-6)
    xp = np.clip(xp * scale, -1.0, 1.0)
    return xp

def to_int16(x: np.ndarray) -> np.ndarray:
    """[-1, 1] の float を int16 へ。事前に normalize_for_saving 済みを想定。"""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x * 32767.0).astype(np.int16)

def write_wav_tmp(sr: int, audio: np.ndarray) -> str:
    """
    一時ファイルにWAVを書き出し、そのパスを返す（DL用）
    - 保存用にだけ normalize を適用
    """
    import wave
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(tempfile.gettempdir(), f"record_{ts}.wav")

    safe = normalize_for_saving(audio, target_peak=0.98)
    audio_i16 = to_int16(safe)

    if audio_i16.ndim == 1:
        n_channels = 1
        frames = audio_i16.tobytes(order="C")
    else:
        n_channels = int(audio_i16.shape[1])
        frames = audio_i16.tobytes(order="C")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(frames)
    return path

def make_wave_plot(y: np.ndarray, title="音声波形", max_points=10_000):
    if y is None or len(y) == 0:
        y = np.array([0.0])
    if len(y) > max_points:
        step = max(1, len(y) // max_points)
        y = y[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y))), y=y.tolist(), mode="lines", name="波形"))
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="振幅",
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig

def make_radar(values, labels, title="音声特性レーダーチャート"):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values or [], theta=labels or [], fill="toself", name="音声特性"))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    return fig

def make_spectrogram_plot(
    x: np.ndarray,
    sr: int,
    title="スペクトログラム（縦=周波数 / 横=時間）",
    n_fft: int = 1024,
    hop: int = 512,
    max_seconds: int = 10,
):
    if x is None or len(x) == 0 or sr <= 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=360, margin=dict(l=20, r=20, t=40, b=40))
        return fig
    if len(x) > sr * max_seconds:
        x = x[: sr * max_seconds]
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames <= 0:
        x = np.pad(x, (0, max(0, n_fft - len(x))))
        n_frames = 1
    spec_mag = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * win
        fft = np.fft.rfft(frame)
        spec_mag.append(np.abs(fft))
    spec_mag = np.array(spec_mag, dtype=np.float32).T
    spec_db = 20.0 * np.log10(spec_mag + 1e-8)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(spec_db.shape[1]) * (hop / sr)
    fig = go.Figure(data=go.Heatmap(z=spec_db, x=times, y=freqs, coloraxis="coloraxis"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="時間 [s]"),
        yaxis=dict(title="周波数 [Hz]"),
        coloraxis=dict(colorbar=dict(title="dB")),
        autosize=True,
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
    )
    return fig

# ─────────────────────────────────────────────────────
# 各グラフの評価コメント
# ─────────────────────────────────────────────────────
def analyze_spectrum_for_comment(x: np.ndarray, sr: int):
    """
    スペクトログラム用のざっくり評価用メトリクスを作成
    - 低音 / 中音 / 高音のエネルギー比
    """
    if x is None or x.size == 0 or sr <= 0:
        return {
            "duration": 0.0,
            "ratio_low": 0.0,
            "ratio_mid": 0.0,
            "ratio_high": 0.0,
        }

    max_seconds = 10
    if x.size > sr * max_seconds:
        x = x[: sr * max_seconds]

    n_fft = 1024
    hop = 512
    win = np.hanning(n_fft).astype(np.float32)

    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    if n_frames <= 0:
        x = np.pad(x, (0, max(0, n_fft - len(x))))
        n_frames = 1

    spec_mag = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * win
        fft = np.fft.rfft(frame)
        spec_mag.append(np.abs(fft))

    spec_mag = np.array(spec_mag, dtype=np.float32).T  # [freq, time]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    total = float(spec_mag.sum() + 1e-12)
    low_band = spec_mag[freqs < 300].sum()
    mid_band = spec_mag[(freqs >= 300) & (freqs < 3000)].sum()
    high_band = spec_mag[freqs >= 3000].sum()

    return {
        "duration": float(len(x) / sr),
        "ratio_low": float(low_band / total),
        "ratio_mid": float(mid_band / total),
        "ratio_high": float(high_band / total),
    }

# ─────────────────────────────────────────────────────
# アプリ本体
# ─────────────────────────────────────────────────────
def create_voice_analysis_app():
    analyzer = SpeechAnalyzer()

    def on_audio_change(audio, auto_tune, ui_silence_thresh, ui_clip_level,
                        progress=gr.Progress(track_tqdm=False)):
        """
        録音受信 → 1) プレビュー即返し 2) 一時WAV（安全正規化） 3) 標準解析を表示
        4) YAMLテンプレで追加フィードバックを追記
        5) LLMへ渡すメトリクスを state に格納（後段イベントで追記）
        """
        empty_fig = make_wave_plot(np.array([0.0]), title="音声波形")
        empty_radar = make_radar([], [])
        empty_spec = make_spectrogram_plot(np.array([0.0]), 16000)

        if audio is None:
            return (
                "音声がありません。マイクで録音してください。",  # 1: result_md
                make_wave_plot(np.array([0.0]), title="録音プレビュー"),  # 2: preview_plot
                empty_fig,   # 3: wave_plot
                "録音すると、ここに波形の解説が表示されます。",  # 4: wave_comment_md
                empty_radar, # 5: radar_plot
                "録音すると、ここにレーダーチャートの解説が表示されます。",  # 6
                empty_spec,  # 7: spectrogram_plot
                "録音すると、ここに声の高さ・響きのコメントが表示されます。",  # 8
                gr.update(value=None, visible=False),  # 9: download_btn
                "",          # 10: llm_state
                "",          # 11: voicy_today_md
            )

        # ---- numpy化＆プレビュー（元信号：解析用にそのまま）----
        if isinstance(audio, tuple) and len(audio) == 2:
            sr, x = audio
            x = np.array(x, dtype=np.float32)
        else:
            sr = 16000
            x = np.array(audio, dtype=np.float32)

        preview_fig = make_wave_plot(
            x[: min(len(x), 2000)],
            title="録音プレビュー（先頭~2000サンプル）"
        )

        # ---- 一時WAV（安全正規化して保存・DLボタン更新）----
        try:
            wav_path = write_wav_tmp(sr, x)  # 内部で normalize_for_saving を適用
            dl_update = gr.update(value=wav_path, visible=True)
        except Exception:
            dl_update = gr.update(value=None, visible=False)

        # ---- しきい値（自動/手動）----
        probe_len = min(len(x), int(sr * 0.5))
        noise_floor = float(np.median(np.abs(x[:probe_len]))) if probe_len > 0 else 0.005
        auto_silence = max(0.01, min(0.08, noise_floor * 3.0))
        auto_clip = 0.98
        silence_thresh = float(auto_silence if auto_tune else ui_silence_thresh)
        clip_level = float(auto_clip if auto_tune else ui_clip_level)

        # ---- 指標（原音ベース）----
        progress(0.25, desc="解析を開始…")
        x_f32 = x.astype(np.float32)

        # 入力レンジの自動判定と正規化
        peak_raw = float(np.max(np.abs(x_f32))) if x_f32.size else 0.0
        if peak_raw > 1.5:
            x_f32 = x_f32 / 32768.0
            peak_raw = float(np.max(np.abs(x_f32)))
        if peak_raw > 1.0:
            x_f32 = x_f32 / peak_raw
            peak_raw = 1.0

        peak = float(np.max(np.abs(x_f32))) if x_f32.size else 0.0
        rms = float(np.sqrt(np.mean(x_f32**2))) if x_f32.size else 0.0
        dbfs = 20.0 * np.log10(max(rms, 1e-12))
        clip_ratio = float((np.abs(x_f32) > clip_level).mean()) if x_f32.size else 0.0
        silence_ratio = float((np.abs(x_f32) < silence_thresh).mean()) if x_f32.size else 1.0
        crest_factor = (peak / (rms + 1e-12)) if rms > 0 else math.inf

        # ---- 特徴抽出用に正規化コピー（解析の安定化）----
        x_for_features = (x_f32 / peak) if peak > 0 else x_f32
        if len(x_for_features) > sr * 60:
            x_for_features = x_for_features[: sr * 60]

        progress(0.5, desc="特徴量を抽出…")
        time.sleep(0.02)

        # ---- 標準解析 ----
        try:
            original_generate_spectrogram = analyzer._generate_spectrogram

            def dummy_generate_spectrogram(*args, **kwargs):
                return {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": [[0, 0], [0, 0]],
                    "min_value": 0,
                    "max_value": 1,
                }

            analyzer._generate_spectrogram = dummy_generate_spectrogram
        except Exception:
            original_generate_spectrogram = None

        try:
            base_result = analyzer.analyze_speech(x_for_features)
        except Exception as e:
            base_result = {
                "feedback": {
                    "良い点": [],
                    "改善点": [],
                    "総合評価": f"解析できませんでした（{e}）。",
                    "アドバイス": [],
                },
                "features": {"速さ": 0, "抑揚": 0, "音量": 0, "明瞭さ": 0, "間": 0},
                "visualization": {
                    "波形": {"x": [0], "y": [0]},
                    "メトリクス": {
                        "values": [0, 0, 0, 0, 0],
                        "labels": ["速さ", "抑揚", "音量", "明瞭さ", "間"],
                    },
                },
            }
        finally:
            if original_generate_spectrogram is not None:
                analyzer._generate_spectrogram = original_generate_spectrogram

        progress(0.8, desc="可視化を描画…")
        time.sleep(0.02)

        # ---- 標準解析の表示整形 ----
        fb = base_result.get("feedback", {})
        good_list = fb.get("良い点", [])
        bad_list = fb.get("改善点", [])
        adv_list = fb.get("アドバイス", [])

        good = "\n".join([f"- {v}" for v in good_list]) or "今日はまだ大きな長所は拾えなかったけど、ここから一緒に育てていこうね。"
        bad = "\n".join([f"- {v}" for v in bad_list]) or "大きな課題は特にないかな。まずは気楽に、しゃべることを楽しんでみて。"
        adv = "\n".join([f"- {v}" for v in adv_list]) or "- 今日はまず『録ることに慣れる』を目標にしてみてね。"

        summary_md = f"""

## トヨコのひとこと総評 💌
{fb.get("総合評価", "今日は声の調子を一緒にチェックしてみたよ。まずは録ってくれてありがとう！")}

## 今日のあなたの“ステキポイント”✨
まずは褒めポイントからいくよ〜。ここはぜひ自信を持ってほしいところね。

{good}

## もうちょっと伸びしろがありそうなところ🌱
ダメ出しじゃなくて、「ここを整えると一気に聞きやすくなるよ〜」というポイントたち。

{bad}

## トヨコからの実践アドバイス🎙
今からでもすぐ試せる、小さなコツをまとめたよ。次に録るとき、どれか1つだけでも意識してみてね。

{adv}

---

## あなたの声を “数字で見える化”（原音ベース）
「感覚」だけじゃなくて、「数字」で見るとこんな感じだよ。

- ピーク: {peak:.3f}（1.0に近いとかなり大きめの声）
- RMS: {rms:.4f} (平均音量、0.03〜0.07 前後が長く聞いても疲れない音量感)
- dBFS: {dbfs:.1f} dBFS（0が最大、-25〜-15 dBFSくらいが心地よい目安）
- クリッピング率（しきい値 {clip_level:.3f}）: {clip_ratio*100:.2f} %（声が割れちゃった割合、0% に近いほど上手にコントロールできてる証拠）
- 無音率（しきい値 {silence_thresh:.3f}）: {silence_ratio*100:.1f} %（間の多さ、一般的に 40〜70% くらいが“呼吸と間”のバランスが良い）
- クレストファクタ: {crest_factor:.2f}（声の鋭さ、ふつうは 3〜15 あたり。20 を超えると山が鋭く、抑揚が強めの傾向）

"""

#---

### 解析時の設定
#- 無音しきい値: {silence_thresh:.3f}
#- クリップ判定レベル: {clip_level:.3f}
#- しきい値モード: {"自動" if auto_tune else "手動"}
#"""

        # グラフ用コメントを生成（都代子さんトーン）
        graph_comments_md = build_graph_comments(
            base_result=base_result,
            peak=peak,
            rms=rms,
            dbfs=dbfs,
            clip_ratio=clip_ratio,
            silence_ratio=silence_ratio,
            crest_factor=crest_factor,
            x_for_spec=x_f32,
            sr=sr,
        )
        # 上の build_graph_comments を
        # 「波形コメント / レーダーコメント / スペクトログラムコメント」
        # に分ける
        wave_comment_text, radar_comment_text, spec_comment_text = graph_comments_md

        # YAML のテンプレ
        yaml_metrics = {
            "dbfs": dbfs,
            "clip_ratio": clip_ratio,
            "silence_ratio": silence_ratio,
            "crest_factor": crest_factor,
            "rms": rms,
            "peak": peak,
        }
        yaml_feedback_md = render_rule_based_feedback(yaml_metrics)
        summary_md += yaml_feedback_md

        # Voicy「今日のあなたに」テキスト
        voicy_section_md = build_voicy_section(base_result.get("features", {}), dbfs)

        # 図
        vis = base_result.get("visualization", {})
        wave = vis.get("波形", {"y": []})
        metrics = vis.get("メトリクス", {"values": [], "labels": []})
        wave_plot = make_wave_plot(np.array(wave.get("y", [])), title="音声波形（解析後）")
        radar_plot = make_radar(metrics.get("values", []), metrics.get("labels", []))
        spec_plot = make_spectrogram_plot(x_f32, sr)

        # —— LLM に渡す payload を State へ（後段で追記）
        llm_metrics = {
            "dbfs": dbfs,
            "clip_ratio": clip_ratio,
            "silence_ratio": silence_ratio,
            "crest_factor": crest_factor,
            "rms": rms,
            "peak": peak,
            "notes": {
                "target_dbfs_range": "[-25, -15]",
                "clip_threshold": clip_level,
                "silence_threshold": silence_thresh,
            },
        }
        llm_state_str = json.dumps(llm_metrics, ensure_ascii=False)

        return (
            summary_md,              # 1: 解析結果まとめ
            preview_fig,             # 2: 録音プレビュー
            wave_plot,               # 3: 波形グラフ
            wave_comment_text,       # 4: 波形コメント
            radar_plot,              # 5: レーダーグラフ
            radar_comment_text,      # 6: レーダーコメント
            spec_plot,               # 7: スペクトログラム
            spec_comment_text,       # 8: スペクトログラムコメント
            dl_update,               # 9: ダウンロードボタン
            llm_state_str,           # 10: llm_state
            voicy_section_md,        # 11: 今日のあなたに 🎧
        )

    def append_llm_feedback(current_md, llm_state_str):
        """LLMで追加フィードバックを生成し追記（API未設定時はそのまま返す）"""

        # ★ ここで環境変数が空ならログに出して抜ける
        #if not (MYGPT_API_BASE and MYGPT_API_KEY and MYGPT_MODEL_ID):
        #    print("[DEBUG] LLM disabled because ENV is missing.")
        #    return current_md

        try:
            metrics = json.loads(llm_state_str or "{}")
        except Exception as e:
            print("[DEBUG] json load error in append_llm_feedback:", e)
            metrics = {}

        system, user = build_llm_prompts(metrics)
        llm_text = call_mygpt(system, user, timeout=8.0)

        if not llm_text:
            print("[DEBUG] LLM returned empty text.")
            return current_md

        section = f"""

---

## 追加フィードバック（AI）
{llm_text}
"""
        return current_md + section

    def reset_all():
        # 空のグラフを作成
        empty_wave = make_wave_plot(np.array([0.0]), title="音声波形")
        empty_radar = make_radar([], [])
        empty_spec = make_spectrogram_plot(np.array([0.0]), 16000)

        return (
            gr.update(value=None),  # 1. audio
            gr.update(              # 2. result_md（解析結果）
                value="録音してください。解析が終わるとグラフが表示されます。"
            ),
            gr.update(              # 3. preview_plot
                value=make_wave_plot(
                    np.array([0.0]), title="録音プレビュー"
                )
            ),
            gr.update(value=empty_wave),  # 4. wave_plot
            gr.update(value=""),          # 5. wave_comment_md
            gr.update(value=empty_radar), # 6. radar_plot
            gr.update(value=""),          # 7. radar_comment_md
            gr.update(value=empty_spec),  # 8. spectrogram_plot
            gr.update(value=""),          # 9. spec_comment_md
            gr.update(value=None, visible=False),  # 10. download_btn
            gr.update(value=""),          # 11. llm_state
            gr.update(                    # voicy_md
                value=(
                    "### 今日のあなたに 🎧 Voicy から\n\n"
                    "解析が終わると、ここにトヨコおすすめの Voicy 放送が表示されます。"
                )
            ),
        )

    # ───────────── UI ─────────────
    with gr.Blocks(
        title="下間都代子の声とことばラボ🎙✨",
        theme=gr.themes.Soft(),
        analytics_enabled=False,
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
    ) as demo:

        with gr.Row():
          gr.Image(
          value="assets/header.gif",
          show_label=False,
          interactive=False,
          elem_id="hero-image"
          )

        gr.HTML("""
<div class="nav-wrapper">
    <!-- メニュー開閉用の隠しチェックボックス -->
    <input type="checkbox" id="menu-toggle" class="menu-toggle" />

    <!-- 三本線（スマホで表示） -->
    <label for="menu-toggle" class="hamburger">☰</label>

    <!-- PC メニュー -->
    <div class="top-menu">
        <a href="#introduction">声とことばラボとは</a>
        <a href="#how-to-use">使い方ガイド</a>
        <a href="#section-analyze">🎙 声を解析する</a>
        <a href="https://chatgpt.com/g/g-68ca42c3955481918334f95460926b26" target="_blank">
            💬 トヨコGPTsで文章づくり
        </a>
    </div>

    <!-- スマホメニュー -->
    <div class="mobile-menu">
        <a href="#introduction">声とことばラボとは</a>
        <a href="#how-to-use">使い方ガイド</a>
        <a href="#section-analyze">🎙 声を解析する</a>
        <a href="https://chatgpt.com/g/g-68ca42c3955481918334f95460926b26" target="_blank">
            💬 トヨコGPTsで文章づくり
        </a>
    </div>
</div>
        """)

        # ★ deep-voice風の 3 カード Feature エリア（入れる場合）
        gr.HTML("""
        <div class="feature-cards">

          <div class="feature-card">
            <h3>🎙 声のクセが一目でわかる</h3>
            <p>音量・速さ・抑揚・明瞭さ。あなたの声のクセを“グラフ”でやさしく見える化します。</p>
          </div>

          <div class="feature-card">
            <h3>💗 トヨコのひとことアドバイス</h3>
            <p>波形・レーダー・スペクトログラムを読み解いて、今日のあなたの声に合わせたフィードバックをトヨコアプリが答えます。</p>
          </div>

          <div class="feature-card">
            <h3>📝 話したくなる文章も作れる</h3>
            <p>トヨコGPTs とつなげて、声だけでなく話し方・文章の魅力もトータルプロデュース。</p>
          </div>

        </div>
        """)

        gr.HTML('<div id="introduction">')
        # ─────────────────────────────
        # アプリ紹介セクション（ヒーロー）
        # ─────────────────────────────
        gr.Markdown("""
# 声とことばラボとは？

ねぇ、声ってね…  
思っている以上に、その人の“いま”が出るんですよ。

ちょっと疲れているときは、音が沈んだり。  
ワクワクしている日は、声の粒が前のめりになったり。  
でもね、本人は案外、その変化に気づかないものなんです。

声には人柄が現れて、話し方にはその人の人間性が現れます。  
それくらい声って重要で正直なんですよね。  
声を聴いただけでも  
その人が本気で生きているかどうかがわかってしまうんです。  

このアプリはね、そんなあなたの声をそっと受け取って、  
「ここね、すごくいいよ」  
「ここを少し整えると、もっと伝わるね」  
って、まるで横で話を聞きながらアドバイスするように、  
やわらかくお伝えするためにつくりました。

それともうひとつ。  
話したいことがうまく言葉にならない日、ありますよね？  
気持ちはあるのに、言葉が追いつかない日。

そんなときは、  
**“トヨコGPTs” があなたの気持ちをそっとすくって、  
話したくなる文章に整えてくれます。**  
無理しなくて大丈夫。あなたのペースで、ね。


- 🎙 マイクを押すだけで、いまの声をキャッチ  
- 📊 波形やレーダーで“あなたの声の表情”が見える  
- 💗 その日の声に合わせて、あなたへ贈りたい Voicy をセレクト  
- ✍️ トヨコGPTs が、伝えたい想いを“やさしく言葉に”してくれる


声はね、あなたのいちばん素直なパートナーです。  
今日のあなたの声が、少しでも軽やかに、心地よく響きますように。  
さぁ、あなたの声、聴かせてくださいね。
        """)

        # デモ動画（使い方イメージ）
        gr.Markdown("#### アプリ紹介動画（イメージ）🎬")
        with gr.Column(elem_classes="custom-video"):
            gr.Video(
                value="assets/demo.mp4",  # 好きな動画ファイルに差し替えてください
                label="デモ動画",
                autoplay=False,
                loop=True,
            )

        gr.HTML('<div id="how-to-use">')
        # ─────────────────────────────
        # 使い方ガイド（任意で簡単に）
        # ─────────────────────────────
        gr.Markdown(
            """
## 使い方ガイド

1. **録音ボタン**を押して、ふだん通りの声で 10〜30 秒ほど話します。  
2. 録音を止めると、自動で解析が始まり、結果とグラフが表示されます。  
3. 「今日のあなたの”ステキポイント”」「もうちょっと伸びしろがありそうなところ」
   「トヨコからの実践アドバイス」を読みながら、グラフとあわせて確認します。  
4. 必要であれば、WAV をダウンロードして、過去の録音と聞き比べてみましょう。

---
"""
        )
        gr.HTML('</div>')

        gr.HTML('<div id="section-analyze">')

        # 録音・解析エリアの見出し（メニューのリンク先）
        gr.Markdown("## 🎙 声を解析する")

        # ★ 録音エリア（大きく＆きらきら枠で目立たせる）
        with gr.Row():
            with gr.Column(elem_id="rec-wrapper"):

                audio = gr.Audio(
                    label="🎙 音声録音（マイク）",
                    type="numpy",
                    sources=["microphone"],
                    elem_id="audio-input",
                )

        reset_btn = gr.Button("🔁 最初からやり直す", variant="secondary")

        # 詳細設定（環境変数 DETAIL_PANEL_VISIBLE で表示/非表示を切り替え）
        if DETAIL_PANEL_VISIBLE:
            with gr.Accordion(
                "くわしいしきい値の設定（上級者向け）",
                open=False,
                visible=True,
            ):
                auto_tune = gr.Checkbox(
                    value=True,
                    label="自動調整（端末・環境ノイズに合わせる）",
                )
                ui_silence_thresh = gr.Slider(
                    minimum=0.005,
                    maximum=0.08,
                    value=0.02,
                    step=0.001,
                    label="無音判定しきい値（手動）",
                )
                ui_clip_level = gr.Slider(
                    minimum=0.90,
                    maximum=1.00,
                    value=0.98,
                    step=0.001,
                    label="クリッピング判定レベル（手動）",
                )
        else:
            # 詳細設定は非表示だが、ロジック上の入力は必要なのでデフォルト値をステートで持つ
            auto_tune = gr.State(True)
            ui_silence_thresh = gr.State(0.02)
            ui_clip_level = gr.State(0.98)

        with gr.Row():
            download_btn = gr.DownloadButton(
                "💾 録音WAVをダウンロード",
                visible=False,
            )

        preview_plot = gr.Plot(label="録音プレビュー")

        # 全体の解析結果まとめ
        result_md = gr.Markdown(label="解析結果")

        # 中段①：波形
        wave_plot = gr.Plot(label="音声波形")
        wave_comment_md = gr.Markdown(label="波形の読み方 ＋ 今回のボリューム評価")

        # 中段②：レーダー
        radar_plot = gr.Plot(label="音声特性レーダー")
        radar_comment_md = gr.Markdown(label="バランスの解説 ＋ 今回の強み")

        # 中段③：スペクトログラム
        spectrogram_plot = gr.Plot(label="スペクトログラム（周波数×時間）")
        spec_comment_md = gr.Markdown(label="高さ・響きの解説")

        gr.HTML('</div>')

        gr.Image(
        value="assets/toyoko-gpts-banner.gif",
        show_label=False,
        interactive=False,
        elem_id="banner-img"
        )

        # LLM状態（非表示）
        llm_state = gr.State("")

        # 今日のあなたに 🎧 Voicy から（独立ブロック）
        gr.Markdown("---")
        voicy_md = gr.Markdown(
            "### 今日のあなたに 🎧 Voicy から\n\n"
            "解析が終わると、ここにトヨコおすすめの Voicy 放送が表示されます。",
            visible=True,
        )

        # 会長ごあいさつセクション
        gr.Markdown("---")
        with gr.Group():
            gr.Markdown("### 株式会社下間都代子コミュニケーション研究所代表")
            gr.Markdown("### 声の総合プロデューサー")
            gr.Markdown("### 全国うっかり協会会長 ご挨拶 📸")
            with gr.Row():
                gr.Image(
                    value="assets/toyoko.jpeg",  # 会長の写真に差し替え
                    label="全国うっかり協会 会長",
                    show_label=False,
                    height=180,
                )
                gr.Markdown(
                    """
下間都代子 💌
"""
                )

        # アプリ初回ロード時に「最初からやり直す」と同じ状態にする
        demo.load(
            fn=reset_all,
            inputs=None,
            outputs=[
                audio,
                result_md,
                preview_plot,
                wave_plot,
                wave_comment_md,
                radar_plot,
                radar_comment_md,
                spectrogram_plot,
                spec_comment_md,
                download_btn,
                llm_state,
                voicy_md,
            ],
        )

        # 録音が変わったら：標準/テンプレ追記の解析を先に表示し、LLM入力をstateへ
        evt = audio.change(
            on_audio_change,
            inputs=[audio, auto_tune, ui_silence_thresh, ui_clip_level],
            outputs=[
                result_md,         # 解析結果まとめ
                preview_plot,      # 録音プレビュー
                wave_plot,         # 波形グラフ
                wave_comment_md,   # 波形の読み方＋今回のボリューム評価
                radar_plot,        # レーダーチャート
                radar_comment_md,  # バランス解説＋今回の強み
                spectrogram_plot,  # スペクトログラム
                spec_comment_md,   # 高さ・響きの解説
                download_btn,      # WAVダウンロードボタン
                llm_state,         # LLM用 state
                voicy_md,          # Voicyから
            ],
            queue=True,
        )

        # LLM 追記部分（ここはそのままで OK）
        evt.then(
            append_llm_feedback,
            inputs=[result_md, llm_state],
            outputs=[result_md],
        )

        # リセット
        reset_btn.click(
            reset_all,
            inputs=None,
            outputs=[
                audio,
                result_md,
                preview_plot,
                wave_plot,
                wave_comment_md,
                radar_plot,
                radar_comment_md,
                spectrogram_plot,
                spec_comment_md,
                download_btn,
                llm_state,
                voicy_md,
            ],
            cancels=[evt],
        )

    demo.queue()
    return demo


if __name__ == "__main__":
    app = create_voice_analysis_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
    )

