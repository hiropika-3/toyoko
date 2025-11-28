"""音声解析モジュール。

このモジュールは、音声録音の処理、音声の可視化、声の特徴分析、
フィードバックとアドバイスの生成を担当します。
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import scipy.signal


class SpeechAnalyzer:
    """音声を解析するクラス。

    このクラスは、音声録音と処理、音声特徴の抽出と分析、可視化データの生成、
    フィードバック生成を担当します。
    """

    def __init__(self):
        """SpeechAnalyzerのコンストラクタ。"""
        # 音声特性の評価基準
        self.voice_metrics = {
            "速さ": {
                "description": "話すスピード（1分あたりの単語数）",
                "ideal_range": (120, 150),  # 理想的な範囲（WPM）
                "feedback": {
                    "too_slow": "もう少し速く話すと、聞き手の注意を維持しやすくなります。",
                    "too_fast": "少しゆっくり話すと、聞き手が内容を理解しやすくなります。",
                    "good": "話すスピードは適切です。聞き手が内容を理解しやすい速さです。"
                }
            },
            "抑揚": {
                "description": "声のピッチ（高低）の変化",
                "ideal_range": (0.5, 1.5),  # 理想的な標準偏差の範囲
                "feedback": {
                    "too_flat": "もう少し抑揚をつけると、話が生き生きとして聞き手の興味を引きます。",
                    "too_varied": "抑揚が大きすぎると落ち着きがない印象を与えることがあります。少し抑えめにすると良いでしょう。",
                    "good": "抑揚のバランスが良く、聞き手の興味を引く話し方です。"
                }
            },
            "音量": {
                "description": "声の大きさ",
                "ideal_range": (60, 75),  # 理想的なデシベル範囲
                "feedback": {
                    "too_quiet": "もう少し声を大きくすると、自信があり説得力のある印象になります。",
                    "too_loud": "声が大きすぎると威圧感を与えることがあります。少し抑えめにすると良いでしょう。",
                    "good": "声の大きさは適切です。聞き取りやすく、自信のある印象を与えています。"
                }
            },
            "明瞭さ": {
                "description": "発音の明瞭さ",
                "ideal_range": (0.7, 1.0),  # 理想的な明瞭度スコア範囲
                "feedback": {
                    "unclear": "発音をもう少し明確にすると、聞き手が内容を理解しやすくなります。",
                    "good": "発音が明確で、聞き手が内容を理解しやすい話し方です。"
                }
            },
            "間": {
                "description": "話の間の取り方",
                "ideal_range": (0.5, 2.0),  # 理想的な間の長さ（秒）
                "feedback": {
                    "too_few": "重要なポイントで間を取ると、聞き手が内容を消化する時間ができ、印象に残りやすくなります。",
                    "too_many": "間が多すぎると話の流れが途切れる印象を与えることがあります。",
                    "good": "間の取り方が適切で、聞き手が内容を理解しやすい話し方です。"
                }
            }
        }

    def analyze_speech(self, audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """音声を解析します。

        Args:
            audio_data: 解析する音声データ（NumPy配列）

        Returns:
            解析結果の辞書
        """
        # 実際の実装では、audio_dataを使用して音声特性を抽出します
        # このサンプル実装では、ダミーデータを生成します
        
        # ダミーの音声特性データ
        speech_features = self._extract_dummy_features()
        
        # 音声特性の評価
        evaluation = self._evaluate_speech_features(speech_features)
        
        # 可視化データの生成
        visualization_data = self._generate_visualization_data(speech_features)
        
        # スペクトログラムの生成
        if audio_data is not None:
            # 実際の音声データからスペクトログラムを生成
            spectrogram_data = self._generate_spectrogram(audio_data)
            visualization_data["spectrogram"] = spectrogram_data
        else:
            # ダミーのスペクトログラムデータを生成
            visualization_data["spectrogram"] = self._generate_dummy_spectrogram()
        
        # フィードバックの生成
        feedback = self._generate_feedback(evaluation)
        
        return {
            "features": speech_features,
            "evaluation": evaluation,
            "visualization": visualization_data,
            "feedback": feedback
        }

    def _extract_dummy_features(self) -> Dict[str, Any]:
        """ダミーの音声特性データを生成します。

        実際の実装では、音声データから特性を抽出します。
        このダミー実装では、毎回異なる結果を生成します。

        Returns:
            音声特性の辞書
        """
        import random
        import time
        
        # 現在時刻をシードとして使用し、毎回異なる結果を生成
        random.seed(time.time())
        
        # 音声特性のパターンをランダムに選択
        patterns = [
            {  # 早口で抑揚が少ない
                "速さ": random.uniform(160, 180),  # WPM (速い)
                "抑揚": random.uniform(0.3, 0.5),  # ピッチの標準偏差 (低い)
                "音量": random.uniform(60, 70),    # デシベル (普通)
                "明瞭さ": random.uniform(0.6, 0.8), # 明瞭度スコア (普通)
                "間": random.uniform(0.2, 0.4),    # 平均間隔（秒）(短い)
            },
            {  # ゆっくりで抑揚が豊か
                "速さ": random.uniform(100, 120),  # WPM (遅い)
                "抑揚": random.uniform(1.2, 1.8),  # ピッチの標準偏差 (高い)
                "音量": random.uniform(65, 75),    # デシベル (普通〜大きい)
                "明瞭さ": random.uniform(0.8, 1.0), # 明瞭度スコア (高い)
                "間": random.uniform(1.5, 2.5),    # 平均間隔（秒）(長い)
            },
            {  # 声が小さく単調
                "速さ": random.uniform(130, 150),  # WPM (普通)
                "抑揚": random.uniform(0.4, 0.6),  # ピッチの標準偏差 (低い)
                "音量": random.uniform(50, 60),    # デシベル (小さい)
                "明瞭さ": random.uniform(0.5, 0.7), # 明瞭度スコア (低い)
                "間": random.uniform(0.5, 1.0),    # 平均間隔（秒）(普通)
            },
            {  # バランスの取れた話し方
                "速さ": random.uniform(130, 150),  # WPM (普通)
                "抑揚": random.uniform(0.8, 1.2),  # ピッチの標準偏差 (普通)
                "音量": random.uniform(65, 75),    # デシベル (普通〜大きい)
                "明瞭さ": random.uniform(0.8, 1.0), # 明瞭度スコア (高い)
                "間": random.uniform(0.8, 1.5),    # 平均間隔（秒）(普通〜長い)
            },
            {  # 声が大きく抑揚が激しい
                "速さ": random.uniform(140, 160),  # WPM (普通〜速い)
                "抑揚": random.uniform(1.5, 1.8),  # ピッチの標準偏差 (高い)
                "音量": random.uniform(75, 85),    # デシベル (大きい)
                "明瞭さ": random.uniform(0.7, 0.9), # 明瞭度スコア (普通〜高い)
                "間": random.uniform(0.3, 0.8),    # 平均間隔（秒）(短い〜普通)
            }
        ]
        
        # ランダムにパターンを選択
        selected_pattern = random.choice(patterns)
        
        # 持続時間を追加
        selected_pattern["持続時間"] = random.uniform(30, 120)  # 音声の長さ（秒）
        
        # ピッチ変化と音量変化のパターンを生成
        if selected_pattern["抑揚"] < 0.6:  # 抑揚が少ない場合
            # 比較的平坦なピッチ変化
            base_pitch = random.uniform(100, 200)
            selected_pattern["ピッチ変化"] = [base_pitch + random.uniform(-20, 20) for _ in range(50)]
        else:  # 抑揚が豊かな場合
            # 変化の大きいピッチ変化
            selected_pattern["ピッチ変化"] = [random.uniform(80, 250) for _ in range(50)]
        
        # 音量変化のパターン
        if selected_pattern["音量"] < 65:  # 声が小さい場合
            base_volume = random.uniform(50, 60)
            selected_pattern["音量変化"] = [base_volume + random.uniform(-10, 10) for _ in range(50)]
        else:  # 声が普通〜大きい場合
            base_volume = selected_pattern["音量"]
            selected_pattern["音量変化"] = [base_volume + random.uniform(-15, 15) for _ in range(50)]
        
        return selected_pattern

    def _evaluate_speech_features(self, features: Dict[str, Any]) -> Dict[str, str]:
        """音声特性を評価します。

        Args:
            features: 音声特性の辞書

        Returns:
            評価結果の辞書
        """
        evaluation = {}
        
        # 速さの評価
        speed = features["速さ"]
        speed_range = self.voice_metrics["速さ"]["ideal_range"]
        if speed < speed_range[0]:
            evaluation["速さ"] = "too_slow"
        elif speed > speed_range[1]:
            evaluation["速さ"] = "too_fast"
        else:
            evaluation["速さ"] = "good"
        
        # 抑揚の評価
        pitch_variation = features["抑揚"]
        pitch_range = self.voice_metrics["抑揚"]["ideal_range"]
        if pitch_variation < pitch_range[0]:
            evaluation["抑揚"] = "too_flat"
        elif pitch_variation > pitch_range[1]:
            evaluation["抑揚"] = "too_varied"
        else:
            evaluation["抑揚"] = "good"
        
        # 音量の評価
        volume = features["音量"]
        volume_range = self.voice_metrics["音量"]["ideal_range"]
        if volume < volume_range[0]:
            evaluation["音量"] = "too_quiet"
        elif volume > volume_range[1]:
            evaluation["音量"] = "too_loud"
        else:
            evaluation["音量"] = "good"
        
        # 明瞭さの評価
        clarity = features["明瞭さ"]
        clarity_range = self.voice_metrics["明瞭さ"]["ideal_range"]
        if clarity < clarity_range[0]:
            evaluation["明瞭さ"] = "unclear"
        else:
            evaluation["明瞭さ"] = "good"
        
        # 間の評価
        pauses = features["間"]
        pause_range = self.voice_metrics["間"]["ideal_range"]
        if pauses < pause_range[0]:
            evaluation["間"] = "too_few"
        elif pauses > pause_range[1]:
            evaluation["間"] = "too_many"
        else:
            evaluation["間"] = "good"
        
        return evaluation

    def _generate_visualization_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """可視化データを生成します。

        Args:
            features: 音声特性の辞書

        Returns:
            可視化データの辞書
        """
        # 実際の実装では、音声特性から可視化用のデータを生成します
        # このサンプル実装では、ダミーデータを返します
        
        visualization_data = {
            "波形": {
                "x": list(range(len(features["音量変化"]))),
                "y": features["音量変化"]
            },
            "ピッチ": {
                "x": list(range(len(features["ピッチ変化"]))),
                "y": features["ピッチ変化"]
            },
            "メトリクス": {
                "labels": ["速さ", "抑揚", "音量", "明瞭さ", "間"],
                "values": [
                    features["速さ"] / 180,  # 正規化
                    features["抑揚"] / 2.0,
                    features["音量"] / 100,
                    features["明瞭さ"],
                    features["間"] / 3.0
                ]
            }
        }
        
        return visualization_data

    def _generate_spectrogram(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """音声データからスペクトログラムを生成します。

        Args:
            audio_data: 音声データ（NumPy配列）
            sample_rate: サンプリングレート（Hz）

        Returns:
            スペクトログラムデータの辞書
        """
        try:
            # 音声データの正規化
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # スペクトログラムの計算
            frequencies, times, spectrogram = scipy.signal.spectrogram(
                audio_data,
                fs=sample_rate,
                window='hanning',
                nperseg=512,
                noverlap=384,
                detrend=False,
                scaling='density'
            )
            
            # デシベルに変換
            spectrogram = 10 * np.log10(spectrogram + 1e-10)
            
            # 人間の声の周波数範囲に制限（80Hz〜800Hz）
            mask = (frequencies >= 80) & (frequencies <= 800)
            frequencies = frequencies[mask]
            spectrogram = spectrogram[mask, :]
            
            # データを辞書形式で返す
            return {
                "x": times.tolist(),  # 時間軸
                "y": frequencies.tolist(),  # 周波数軸
                "z": spectrogram.tolist(),  # 強度（デシベル）
                "min_value": np.min(spectrogram),
                "max_value": np.max(spectrogram)
            }
        except Exception as e:
            # エラーが発生した場合はダミーデータを返す
            print(f"スペクトログラム生成中にエラーが発生しました: {str(e)}")
            return self._generate_dummy_spectrogram()

    def _generate_dummy_spectrogram(self) -> Dict[str, Any]:
        """ダミーのスペクトログラムデータを生成します。

        Returns:
            スペクトログラムデータの辞書
        """
        import random
        
        # 時間軸と周波数軸の設定
        times = np.linspace(0, 5, 100)  # 0〜5秒
        frequencies = np.linspace(80, 800, 50)  # 80Hz〜800Hz
        
        # スペクトログラムデータの生成
        spectrogram = np.zeros((len(frequencies), len(times)))
        
        # 基本周波数（女性の声を想定: 約200Hz）
        base_freq_idx = np.abs(frequencies - 200).argmin()
        
        # 基本周波数の変動を生成
        for t_idx in range(len(times)):
            # 基本周波数の変動（抑揚）
            variation = 10 * np.sin(2 * np.pi * 0.5 * times[t_idx])
            center_freq_idx = base_freq_idx + int(variation)
            center_freq_idx = max(0, min(center_freq_idx, len(frequencies) - 1))
            
            # 基本周波数とその倍音にエネルギーを与える
            for harmonic in range(1, 4):  # 基本周波数と2倍音、3倍音
                harmonic_idx = min(center_freq_idx * harmonic, len(frequencies) - 1)
                
                # ガウス分布で周辺の周波数にもエネルギーを分散
                for f_idx in range(len(frequencies)):
                    distance = abs(f_idx - harmonic_idx)
                    if distance < 5:  # 近い周波数にのみ影響
                        energy = 30 * np.exp(-0.5 * (distance / 2) ** 2)  # ガウス分布
                        
                        # 時間によるエネルギーの変動（音量）
                        volume_variation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.2 * times[t_idx])
                        
                        spectrogram[f_idx, t_idx] = energy * volume_variation
        
        # 間（無音部分）を追加
        pause_positions = [20, 60]  # 時間軸上の位置
        for pos in pause_positions:
            if pos < len(times) - 5:
                spectrogram[:, pos:pos+5] *= 0.1  # 音量を下げる
        
        # ノイズを追加
        noise = np.random.rand(len(frequencies), len(times)) * 5
        spectrogram += noise
        
        # 値の範囲を調整
        min_value = 0
        max_value = 40
        spectrogram = np.clip(spectrogram, min_value, max_value)
        
        return {
            "x": times.tolist(),  # 時間軸
            "y": frequencies.tolist(),  # 周波数軸
            "z": spectrogram.tolist(),  # 強度（デシベル）
            "min_value": min_value,
            "max_value": max_value
        }

    def _generate_feedback(self, evaluation: Dict[str, str]) -> Dict[str, Any]:
        """フィードバックを生成します。

        Args:
            evaluation: 評価結果の辞書

        Returns:
            フィードバックの辞書
        """
        feedback = {
            "良い点": [],
            "改善点": [],
            "総合評価": "",
            "アドバイス": []
        }
        
        # 良い点と改善点の抽出
        for metric, result in evaluation.items():
            if result == "good":
                feedback["良い点"].append(f"{metric}: {self.voice_metrics[metric]['feedback'][result]}")
            else:
                feedback["改善点"].append(f"{metric}: {self.voice_metrics[metric]['feedback'][result]}")
        
        # 総合評価の生成
        good_points = len(feedback["良い点"])
        total_points = len(evaluation)
        score = good_points / total_points
        
        if score >= 0.8:
            feedback["総合評価"] = "素晴らしい話し方です！聞き手に強い印象を与えることができるでしょう。"
        elif score >= 0.6:
            feedback["総合評価"] = "良い話し方です。いくつかの点を改善すると、さらに効果的になります。"
        elif score >= 0.4:
            feedback["総合評価"] = "基本的な話し方はできています。改善点に取り組むと、より効果的になります。"
        else:
            feedback["総合評価"] = "話し方に改善の余地があります。アドバイスを参考に練習を続けましょう。"
        
        # アドバイスの生成
        if "速さ" in evaluation and evaluation["速さ"] != "good":
            feedback["アドバイス"].append("話すスピードを意識して練習してみましょう。メトロノームを使うと効果的です。")
        
        if "抑揚" in evaluation and evaluation["抑揚"] != "good":
            feedback["アドバイス"].append("感情を込めて話す練習をしましょう。詩や物語の朗読が効果的です。")
        
        if "音量" in evaluation and evaluation["音量"] != "good":
            feedback["アドバイス"].append("適切な声の大きさを意識しましょう。録音して確認すると良いでしょう。")
        
        if "明瞭さ" in evaluation and evaluation["明瞭さ"] != "good":
            feedback["アドバイス"].append("発音練習を行いましょう。早口言葉や滑舌トレーニングが効果的です。")
        
        if "間" in evaluation and evaluation["間"] != "good":
            feedback["アドバイス"].append("重要なポイントで意識的に間を取る練習をしましょう。")
        
        # モチベーションを高めるメッセージ
        feedback["アドバイス"].append("継続的な練習が話し方の上達につながります。自分の声を録音して聞くことで、改善点が見えてきます。")
        
        return feedback

    def record_audio(self, duration: int = 10) -> np.ndarray:
        """音声を録音します。

        実際の実装では、マイクから音声を録音します。
        このサンプル実装では、ダミーデータを返します。

        Args:
            duration: 録音時間（秒）

        Returns:
            録音された音声データ（NumPy配列）
        """
        # 実際の実装では、マイクから音声を録音します
        # このサンプル実装では、ダミーデータを返します
        sample_rate = 16000
        dummy_audio = np.random.rand(duration * sample_rate) * 2 - 1
        return dummy_audio

    def visualize_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """音声を可視化します。

        Args:
            audio_data: 音声データ（NumPy配列）

        Returns:
            可視化データの辞書
        """
        # 波形データ
        waveform = {
            "x": list(range(min(1000, len(audio_data)))),
            "y": audio_data[:1000].tolist() if len(audio_data) >= 1000 else audio_data.tolist()
        }
        
        # スペクトログラムデータ
        spectrogram = self._generate_spectrogram(audio_data)
        
        return {
            "waveform": waveform,
            "spectrogram": spectrogram
        }
