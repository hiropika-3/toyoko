"""
独自解析プラグインのサンプル
-----------------------------

アプリ側からは:
    from plugins.analysis_custom import analyze

が呼ばれます。

入力:
    audio: numpy.ndarray (float32, -1.0〜1.0 正規化想定)
    sr:    int (サンプルレート)

出力:
    dict (必須キー):
        "feedback": {
            "良い点": [str, ...],
            "改善点": [str, ...],
            "総合評価": str,
            "アドバイス": [str, ...],
        },
        "features": {   # 必須だけど中身は適当でもOK
            "速さ": float,
            "抑揚": float,
            "音量": float,
            "明瞭さ": float,
            "間": float,
        },
        "visualization": {   # 波形やレーダーを追加で表示したい場合
            "波形": {"x": [...], "y": [...]},
            "メトリクス": {"values": [...], "labels": [...]},
        }
"""

import numpy as np

def analyze(audio: np.ndarray, sr: int) -> dict:
    # ---- 簡易特徴量抽出 ----
    duration_sec = len(audio) / sr if len(audio) > 0 else 0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    # 「速さ」っぽい指標を発話時間から簡易に
    speed_score = min(1.0, duration_sec / 60.0)  # 1分で1.0

    # 「抑揚」っぽい指標をピーク/RMSから
    modulation_score = min(1.0, (peak - rms) * 5)

    # 「音量」っぽい指標をRMSから
    loudness_score = min(1.0, rms * 10)

    # 「明瞭さ」「間」は仮で定数
    clarity_score = 0.6
    pause_score = 0.5

    # ---- アドバイス生成 ----
    feedback = {
        "良い点": ["声の安定感が感じられます。", "テンポが自然です。"],
        "改善点": ["語尾が少し弱く聞こえる部分があります。"],
        "総合評価": "独自解析: バランスは良好ですが、さらに抑揚を意識すると効果的です。",
        "アドバイス": [
            "キーワードの直前で一呼吸置くとより伝わりやすいです。",
            "強調したい部分は少し声を張ると良いでしょう。"
        ]
    }

    # ---- 可視化用データ（必須ではないが一応） ----
    visualization = {
        "波形": {"x": list(range(len(audio))), "y": audio.tolist()},
        "メトリクス": {
            "values": [speed_score, modulation_score, loudness_score, clarity_score, pause_score],
            "labels": ["速さ", "抑揚", "音量", "明瞭さ", "間"]
        }
    }

    return {
        "feedback": feedback,
        "features": {
            "速さ": speed_score,
            "抑揚": modulation_score,
            "音量": loudness_score,
            "明瞭さ": clarity_score,
            "間": pause_score
        },
        "visualization": visualization
    }
