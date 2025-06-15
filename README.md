# AI-Prompt-Evaluation-Script-Python

**AI Prompt Evaluation** is a Python script that compares responses from two AI models—like ChatGPT and Claude—based on how effective, clear, and helpful their answers are.

It uses four custom metrics inspired by JOJO:

* **Joy**: How positive the tone of the response is
* **Outcomes**: How well the response covers key ideas or keywords you’re looking for
* **Journey**: How readable and smooth the response is
* **Opportunity**: Whether the response offers useful, actionable suggestions

The script also checks the emotional tone using sentiment analysis (polarity and subjectivity), and calculates an **Emotional Resonance** score based on how expressive the language is.

You give it two text files (one from each model) and a list of keywords to look for, and it prints out all the scores so you can see which response is better.
