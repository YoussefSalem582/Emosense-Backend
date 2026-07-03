"""
Emotion label utilities shared across analyzers.

Maps assorted raw model labels (e.g. "happy", "angry") onto the canonical
:class:`~app.models.emotion.EmotionLabel` set and derives the dominant emotion.
"""

from typing import Dict, Tuple

from app.models.emotion import EmotionLabel


# Map common raw labels emitted by various models to our canonical labels.
_RAW_LABEL_MAP = {
    "happy": EmotionLabel.JOY,
    "happiness": EmotionLabel.JOY,
    "joy": EmotionLabel.JOY,
    "sad": EmotionLabel.SADNESS,
    "sadness": EmotionLabel.SADNESS,
    "angry": EmotionLabel.ANGER,
    "anger": EmotionLabel.ANGER,
    "fear": EmotionLabel.FEAR,
    "fearful": EmotionLabel.FEAR,
    "surprise": EmotionLabel.SURPRISE,
    "surprised": EmotionLabel.SURPRISE,
    "disgust": EmotionLabel.DISGUST,
    "disgusted": EmotionLabel.DISGUST,
    "love": EmotionLabel.LOVE,
    "excitement": EmotionLabel.EXCITEMENT,
    "excited": EmotionLabel.EXCITEMENT,
    "neutral": EmotionLabel.NEUTRAL,
    "calm": EmotionLabel.NEUTRAL,
}


def standardize_emotions(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Map arbitrary raw emotion scores onto the canonical label set.

    Unknown labels are ignored. The result always contains every
    :class:`EmotionLabel` value (missing ones default to ``0.0``).
    """
    scores = {label.value: 0.0 for label in EmotionLabel}

    for raw_label, score in (raw_scores or {}).items():
        mapped = _RAW_LABEL_MAP.get(str(raw_label).lower())
        if mapped is not None:
            scores[mapped.value] = max(scores[mapped.value], float(score))

    return scores


def get_dominant_emotion(scores: Dict[str, float]) -> Tuple[EmotionLabel, float]:
    """Return the (dominant emotion, confidence) for a standardized score dict."""
    if not scores:
        return EmotionLabel.NEUTRAL, 0.0

    dominant_value = max(scores.keys(), key=lambda k: scores[k])
    confidence = scores[dominant_value]

    try:
        return EmotionLabel(dominant_value), confidence
    except ValueError:
        return EmotionLabel.NEUTRAL, 0.0
