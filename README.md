

---

# 🎧 YouTube History Music Recommender  
> *내 유튜브 기록을 분석해 음악 추천까지!*  

---

## 🏹 프로젝트 목적  
🎯 **실제 소비 데이터 기반 인사이트 도출**  
- 유튜브 시청 기록 전처리 → 키워드, 시간대, 콘텐츠 유형별 분석  
- 콘텐츠 소비 습관과 감정 흐름 파악

🎶 **음악 추천 시스템 설계**  
- NLP 기반 키워드/감정 분석  
- YouTube Data API로 메타데이터 확보  
- 향후 Spotify API 연동으로 추천 정확도 향상

---

## 🛠️ 기술 스택  

| 분야             | 기술/라이브러리                              |
|------------------|-----------------------------------------------|
| 🔤 언어           | Python 3.10+                                  |
| 📊 데이터 분석    | pandas, numpy, matplotlib, seaborn            |
| 🧠 NLP/감성 분석  | spaCy, konlpy, transformers (KcELECTRA)       |
| 🌐 API            | YouTube Data API *(Spotify API 예정)*        |
| 📈 시각화         | WordCloud, Plotly, Streamlit *(웹 확장 가능)* |

---

## 🌱 프로젝트 구조

```
📂 youtube-history-music-recommender/
├── README.md
├── data/
│   └── youtube_history.csv
├── scripts/
│   ├── 01_preprocess_history_data.py
│   ├── 02_analyze_enriched_data.py
│   └── 03_
```

---

## 🌍 English Version  

### 🎧 YouTube History Music Recommender  
> *Analyzing my watch history to uncover insights and recommend music.*

---

### 🏹 Project Goals  

🎯 **Extract Insights from Real Consumption Data**  
- Process YouTube watch history and analyze viewing patterns  
- Categorize by keyword, time, and content type

🎶 **Build a Music Recommendation System**  
- Perform NLP-based keyword and sentiment analysis  
- Use YouTube Data API to enrich metadata  
- Expand recommendation logic with Spotify API (planned)

---

### 🛠️ Tech Stack  

| Area             | Tools & Libraries                              |
|------------------|--------------------------------------------------|
| 🔤 Language        | Python 3.10+                                     |
| 📊 Data Analysis   | pandas, numpy, matplotlib, seaborn               |
| 🧠 NLP/Sentiment   | spaCy, konlpy, transformers (KcELECTRA)          |
| 🌐 API             | YouTube Data API *(Spotify API planned)*        |
| 📈 Visualization   | WordCloud, Plotly, Streamlit *(for web expansion)* |

---
