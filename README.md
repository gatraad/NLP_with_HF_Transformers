<div align="center">

# **Natural Language Processing with Hugging Face Transformers**
### *Generative AI Guided Project on Cognitive Class by IBM*
  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

</div>

---

## 👤 **Name:** Gatra Adi Wirya  

---

## 🚀 **My Project Tasks**

### 🧩 **Example 1 – Sentiment Analysis**

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I am doing this on a regular basis, baking a cake in the morning!")
````

**Result:**

```python
[{'label': 'POSITIVE', 'score': 0.9959210157394409}]
```

**Analysis:**
The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.

---

### 🏷️ **Example 2 – Topic Classification**

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Cats are beloved domestic companions known for their independence and agility...",
    candidate_labels=["science", "pet", "machine learning"],
)
```

**Result:**

```python
{'labels': ['pet', 'machine learning', 'science'], 'scores': [0.9174, 0.0485, 0.0339]}
```

**Analysis:**
The zero-shot classifier correctly identifies **“pet”** as the most relevant label with high confidence. This demonstrates the model’s ability to associate context with appropriate categories even without specific training for the task.

---

### ✍️ **Example 3 & 3.5 – Text Generation & Fill Mask**

```python
generator = pipeline("text-generation", model="distilgpt2")
generator("This cooking will make you", max_length=30, num_return_sequences=2)
```

**Result:**

```python
"This cooking will make you feel good and warm. It will also take time..."
```

**Analysis:**
The text generation model produces coherent and creative continuations of the cooking-themed prompt. It maintains smooth sentence flow and relevance, though some repetitions appear. Overall, it highlights the model’s strength in generating natural, engaging text.

---

```python
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("This person is the one who <mask> my purse", top_k=4)
```

**Result:**

```python
[{'token_str': ' stole', 'score': 0.8569}, {'token_str': ' snatched', 'score': 0.0309}, ...]
```

**Analysis:**
The fill-mask pipeline accurately infers the missing word based on sentence context. The top prediction **“stole”** fits perfectly, supported by a high confidence score. Other suggestions remain contextually logical, showing the model’s strong language understanding.

---

### 🧭 **Example 4 – Named Entity Recognition (NER)**

```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Gatra Adi Wirya, I am an AI Student at Infinite Learning, Palembang City")
```

**Result:**

```python
[{'entity_group': 'PER', 'word': 'Gatra Adi Wirya'},
 {'entity_group': 'ORG', 'word': 'Infinite Learning'},
 {'entity_group': 'LOC', 'word': 'Palembang City'}]
```

**Analysis:**
The named entity recognizer successfully identifies names, organizations, and locations with high confidence. This highlights its effectiveness for real-world tasks like document tagging and information extraction.

---

### ❓ **Example 5 – Question Answering**

```python
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What four-legged animal sometimes comes inside the house and likes to sleep?"
context = "Four-legged animal that sometimes comes inside the house and likes to sleep is a cat"
qa_model(question=question, context=context)
```

**Result:**

```python
{'answer': 'a cat', 'score': 0.6314}
```

**Analysis:**
The question-answering model correctly extracts **“a cat”** as the most relevant answer. It demonstrates solid comprehension of natural questions and effective span extraction from the given context.

---

### 📰 **Example 6 – Text Summarization**

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer("""
Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan sistem komputer untuk belajar dari data tanpa diprogram secara eksplisit...
""")
```

**Result:**

```python
"Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit."
```

**Analysis:**
The summarization pipeline effectively condenses the main ideas while preserving essential information. It captures key points such as learning from data and pattern recognition, showing good comprehension of text structure.

---

### 🌍 **Example 7 – Translation**

```python
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
```

**Result:**

```python
[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]
```

**Analysis:**
The translation model accurately converts the Indonesian sentence into French while maintaining tone and meaning. It performs well in casual conversational contexts, demonstrating flexibility in multilingual understanding.

---

## 📊 **Overall Project Analysis**

This project provides a hands-on exploration of diverse NLP tasks using Hugging Face pipelines. Each example demonstrates practical use cases—from sentiment detection to summarization—highlighting the adaptability and power of transformer-based models in real-world language applications.

```
