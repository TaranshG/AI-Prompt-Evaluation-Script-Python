#!/usr/bin/env python3
"""
AI Prompt Evaluation Script

Compares outputs from two AI models (e.g., ChatGPT and Claude)
using custom JOJO-inspired metrics:
  - Joy: sentiment positivity
  - Outcomes: coverage of expected key points
  - Journey: readability / flow
  - Opportunity: actionable suggestions

Also reports:
  - Polarity & Subjectivity (tone analysis)
  - Emotional Resonance (from subjectivity)

Usage:
  python ai_prompt_evaluator.py \
    --chatgpt_path chatgpt_out.txt \
    --claude_path claude_out.txt \
    --ref_points "joy metric,flow,actionable suggestions"

Dependencies:
  pip install textblob textstat
  python -m textblob.download_corpora
"""

import argparse                  # For command-line argument parsing
from textblob import TextBlob    # For sentiment (polarity & subjectivity)
import textstat                  # For readability (Flesch Reading Ease)

def score_joy(text):
    """
    Joy metric:
      - Uses TextBlob polarity (range -1 to +1)
      - Maps to 0–10 via (polarity + 1) * 5
    """
    polarity = TextBlob(text).sentiment.polarity
    return (polarity + 1.0) * 5.0

def score_outcomes(text, refs):
    """
    Outcomes metric:
      - refs: list of keywords/phrases to check for coverage
      - Counts how many refs appear in text
      - Scales proportion found to 0–10
    """
    if not refs:
        return 0.0
    text_lower = text.lower()
    count = sum(1 for r in refs if r.lower() in text_lower)
    return (count / len(refs)) * 10.0

def score_journey(text):
    """
    Journey metric (readability):
      - Uses Flesch Reading Ease score (0–100) from textstat
      - Clamps to [0,100] then maps to 0–10 by dividing by 10
    """
    ease = textstat.flesch_reading_ease(text)
    ease = max(0.0, min(ease, 100.0))
    return ease / 10.0

def score_opportunity(text):
    """
    Opportunity metric:
      - Looks for actionable words: recommend, you can, consider, try, suggest
      - Counts occurrences, scales (count/5)*10, caps at 10
    """
    markers = ["recommend", "you can", "consider", "try", "suggest"]
    text_lower = text.lower()
    total = sum(text_lower.count(m) for m in markers)
    score = (total / 5.0) * 10.0
    return score if score <= 10.0 else 10.0

def analyze_tone(text):
    """
    Tone analysis:
      - Returns polarity and subjectivity (each 0–1) from TextBlob
    """
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity

def evaluate(text, refs):
    """
    Runs all JOJO metrics + tone analysis on a single text.
    Returns a dict with rounded scores.
    """
    joy = round(score_joy(text), 2)
    outcomes = round(score_outcomes(text, refs), 2)
    journey = round(score_journey(text), 2)
    opportunity = round(score_opportunity(text), 2)
    polarity, subjectivity = analyze_tone(text)
    emotional_resonance = round(subjectivity * 10.0, 2)

    return {
        "Joy": joy,
        "Outcomes": outcomes,
        "Journey": journey,
        "Opportunity": opportunity,
        "Polarity": round(polarity, 2),
        "Subjectivity": round(subjectivity, 2),
        "Emotional Resonance": emotional_resonance
    }

def load_text(path):
    """
    Loads the entire contents of a text file and returns it as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    """
    Main script flow:
      - Parse CLI args
      - Load each model’s output
      - Evaluate with JOJO metrics
      - Print results
    """
    parser = argparse.ArgumentParser(
        description="Compare two AI outputs using JOJO metrics"
    )
    parser.add_argument("--chatgpt_path", required=True,
                        help="Path to ChatGPT output file")
    parser.add_argument("--claude_path", required=True,
                        help="Path to Claude output file")
    parser.add_argument("--ref_points", default="",
                        help="Comma-separated list of reference keywords")
    args = parser.parse_args()

    # Prepare list of reference keywords
    refs = [r.strip() for r in args.ref_points.split(",") if r.strip()]

    # Read model outputs
    chatgpt_text = load_text(args.chatgpt_path)
    claude_text = load_text(args.claude_path)

    # Evaluate ChatGPT output
    print("=== ChatGPT Evaluation ===")
    for metric, value in evaluate(chatgpt_text, refs).items():
        print(f"{metric}: {value}")

    # Evaluate Claude output
    print("\n=== Claude Evaluation ===")
    for metric, value in evaluate(claude_text, refs).items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
