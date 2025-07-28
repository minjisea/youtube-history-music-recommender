# ---------- 완전한 데이터 보강 및 통합 ----------
import pandas as pd, numpy as np, re, json, datetime as dt, pathlib, textwrap, math
from collections import Counter
import seaborn as sns, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from googleapiclient.discovery import build  # YouTube Data API v3

# ---------- 1.  CONFIG ----------
DATA_PATH        = pathlib.Path("C:/Users/ohdon/Downloads/시청기록최근3년.csv")   # adjust
API_KEY          = ""                 # optional but unlocks richer insights
MAX_API_BATCH    = 50                                  # YT API limit
SESSION_GAP      = pd.Timedelta(minutes=30)            # gap that splits sessions

# ---------- 2.  LOAD & CLEAN ----------
df = (pd.read_csv(DATA_PATH)
        .rename(columns=str.lower)
        .assign(watched_at = lambda d: pd.to_datetime(d['timestamp']))
        .dropna(subset=['title','url','watched_at'])
     )
df['video_id']   = df['url'].str.extract(r"v=([\w-]{11})")
df['weekday']    = df['watched_at'].dt.day_name()
df['hour']       = df['watched_at'].dt.hour
df = df.sort_values('watched_at').reset_index(drop=True)
df['minutes'] = df.get('duration', pd.Series([0]*len(df))).fillna(0)

# ---------- 2-A. 세션 구간 계산 ----------
# 직전 시청 시각
df['prev_time']   = df['watched_at'].shift()
# 30분 이상 끊기면 새 세션
df['new_session'] = (df['watched_at'] - df['prev_time'] > SESSION_GAP) | df['prev_time'].isna()
# 세션 ID 부여
df['session_id']  = df['new_session'].cumsum()

# 세션별 통계 집계
session_stats = (df.groupby('session_id')
                   .agg(session_start=('watched_at','min'),
                        session_end  =('watched_at','max'),
                        session_videos=('url','count'))
                   .assign(session_duration_minutes=
                           lambda t: (t['session_end']-t['session_start'])
                                     .dt.total_seconds()/60)
                   .reset_index())

# 원본 df에 병합
df = df.merge(session_stats, on='session_id', how='left')

# ---------- 2. YouTube API로 메타데이터 보강 ----------
def fetch_meta(batch_ids: list) -> pd.DataFrame:
    """YouTube Data API v3로 duration, channel, category 가져오기"""
    try:
        yt = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)
        resp = (yt.videos()
                  .list(id=",".join(batch_ids),
                        part="snippet,contentDetails")
                  .execute())
        items = []
        for v in resp["items"]:
            # ISO 8601 duration을 분으로 변환
            duration_str = v["contentDetails"]["duration"]
            duration_match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
            hours = int(duration_match.group(1) or 0)
            minutes = int(duration_match.group(2) or 0) 
            seconds = int(duration_match.group(3) or 0)
            total_minutes = hours * 60 + minutes + seconds / 60
            
            items.append({
                "video_id": v["id"],
                "channel_title": v["snippet"]["channelTitle"],
                "category_id": v["snippet"]["categoryId"],
                "duration": total_minutes,
                "published_at": v["snippet"]["publishedAt"]
            })
        return pd.DataFrame(items)
    except Exception as e:
        print(f"API 호출 실패: {e}")
        return pd.DataFrame()

# API 키가 설정되어 있으면 메타데이터 가져오기
if API_KEY != "API-KEY":
    unique_video_ids = df["video_id"].dropna().unique()
    meta_frames = []
    
    print(f"video_id 개수: {len(unique_video_ids)}")
    print(unique_video_ids[:5])  # 샘플 확인
    # 배치 단위로 API 호출
    for i in range(0, len(unique_video_ids), MAX_API_BATCH):
        batch = unique_video_ids[i:i+MAX_API_BATCH]
        meta_batch = fetch_meta(list(batch))
        if not meta_batch.empty:
            meta_frames.append(meta_batch)
    
    if meta_frames:
        meta = pd.concat(meta_frames, ignore_index=True)
        # df에 메타데이터 병합 (duration 덮어쓰기)
        df = df.merge(meta, on="video_id", how="left")
        # minutes 컬럼을 duration으로 업데이트
        df['minutes'] = df['duration'].fillna(df['minutes'])
        df['duration'] = df['minutes']
        print(f"✅ {len(meta)} 개 영상의 메타데이터 추가 완료")
    else:
        print("⚠️ API 메타데이터 가져오기 실패 - 기존 데이터 사용")
        df['channel_title'] = 'Unknown'
        df['category_id'] = 'Unknown'
        df['duration'] = df['minutes']
else:
    print("⚠️ API 키가 설정되지 않음 - 추정값 사용")
    # API 없이 추정값으로 컬럼 채우기
    df['channel_title'] = df['title'].str.split(' ').str[0]  # 첫 단어를 채널명으로 추정
    df['category_id'] = df['topic'].astype(str)  # 토픽을 카테고리로 사용
    df['duration'] = df['minutes']

# ---------- 3. 추가 파생 컬럼 생성 ----------
# 시간 관련
df['date'] = df['watched_at'].dt.date
df['month'] = df['watched_at'].dt.month
df['year'] = df['watched_at'].dt.year
df['season'] = df['month'].apply(lambda x: 
    'Spring' if x in [3,4,5] else
    'Summer' if x in [6,7,8] else  
    'Fall' if x in [9,10,11] else 'Winter')

df['time_period'] = df['hour'].apply(lambda x:
    'Dawn' if 0 <= x < 6 else
    'Morning' if 6 <= x < 12 else
    'Afternoon' if 12 <= x < 18 else 'Evening')

df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday'])

# 영상 길이 카테고리
df['duration_category'] = df['duration'].apply(lambda x:
    'Short' if x < 4 else
    'Medium' if x < 20 else 'Long')

# 몰아보기 세션 식별 (3개 이상 연속 시청)
df['is_binge_session'] = df['session_videos'] >= 3

# 같은 채널 연속 시청 감지
df['prev_channel'] = df.groupby('session_id')['channel_title'].shift(1)
df['is_channel_binge'] = (df['channel_title'] == df['prev_channel'])

print("✅ 모든 컬럼 생성 완료!")
print(f"현재 df 컬럼 수: {len(df.columns)}")
print("주요 컬럼들:", [col for col in df.columns if col in ['duration', 'channel_title', 'category_id', 'session_duration_minutes', 'session_videos']])

# ---------- 3. 텍스트 전처리 및 토픽 클러스터링 ----------
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 제목 클린징
def keyword_clean(text):
    return re.sub(r'[^\w\s]', '', text).lower()

df['clean_title'] = df['title'].astype(str).apply(keyword_clean)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df['clean_title'])

# K-Means 클러스터링(예: 8개 클러스터)
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['topic'] = kmeans.fit_predict(tfidf_matrix)

print("✅ 토픽 클러스터링 완료 — topic 컬럼이 생성되었습니다.")


# ---------- 4. 완전한 데이터 저장 ----------
df.to_csv("C:/Users/ohdon/Downloads/ytEnrichData_complete.csv", index=False, encoding='utf-8-sig')
print("✅ 완전한 데이터프레임 저장 완료")

# ---------- 5. 집계 테이블 CSV 저장 ----------
# 5-1) 요일·시간대별 시청 빈도
hourly_heatmap = (df.groupby(['weekday', 'hour'])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(['Monday','Tuesday','Wednesday',
                              'Thursday','Friday','Saturday','Sunday']))
hourly_heatmap.to_csv("C:/Users/ohdon/Downloads/watch_heatmap_weekday_hour.csv")

# 5-2) 채널 TOP 30
top_channels = (df.groupby('channel_title')
                  .agg(videos=('video_id','count'),
                       minutes=('duration','sum'),
                       avg_duration=('duration','mean'))
                  .sort_values('videos', ascending=False)
                  .head(30)
                  .reset_index())
top_channels.to_csv("C:/Users/ohdon/Downloads/top30_channels.csv", index=False)

# 5-3) 카테고리별 시청 시간
category_watch = (df.groupby('category_id')
                    .agg(videos=('video_id','count'),
                         minutes=('duration','sum'))
                    .sort_values('minutes', ascending=False)
                    .reset_index())
category_watch.to_csv("C:/Users/ohdon/Downloads/watch_by_category.csv", index=False)

# 5-4) 토픽 클러스터별 통계
topic_summary = (df.groupby('topic')
                   .agg(videos=('video_id','count'),
                        minutes=('duration','sum'))
                   .sort_values('videos', ascending=False)
                   .reset_index())
topic_summary.to_csv("C:/Users/ohdon/Downloads/topic_cluster_summary.csv", index=False)

# 5-5) 세션별 상세 통계
session_detail = (df.groupby('session_id')
                    .agg(videos=('video_id','count'),
                         duration_minutes=('session_duration_minutes','first'),
                         total_watch_minutes=('duration','sum'),
                         start_time=('watched_at','min'),
                         end_time=('watched_at','max'),
                         is_binge=('is_binge_session','first'))
                    .reset_index())
session_detail.to_csv("C:/Users/ohdon/Downloads/session_detailed_stats.csv", index=False)

# 5-6) 일별 시청 통계
daily_stats = (df.groupby('date')
                 .agg(videos=('video_id','count'),
                      minutes=('duration','sum'),
                      sessions=('session_id','nunique'))
                 .reset_index())
daily_stats.to_csv("C:/Users/ohdon/Downloads/daily_stats.csv", index=False)

# 5-7) 몰아보기 분석
binge_analysis = (df[df['is_binge_session']]
                    .groupby('session_id')
                    .agg(videos=('video_id','count'),
                         duration=('session_duration_minutes','first'),
                         watch_minutes=('duration','sum'))
                    .reset_index())
binge_analysis.to_csv("C:/Users/ohdon/Downloads/binge_sessions.csv", index=False)

print("✅ 모든 집계 CSV 저장 완료")

# ---------- 6. 시각화 ----------
plt.style.use('seaborn-v0_8')

# 6-1) 요일·시간대 히트맵
plt.figure(figsize=(12,6))
sns.heatmap(hourly_heatmap, cmap="YlOrRd", linewidths=.5, annot=False)
plt.title("Hourly Watch Frequency by Weekday")
plt.savefig("C:/Users/ohdon/Downloads/heatmap_weekday_hour.png", dpi=300, bbox_inches='tight')
plt.close()

# 6-2) 상위 10개 채널 막대그래프
plt.figure(figsize=(12,8))
sns.barplot(data=top_channels.head(10),
            y='channel_title', x='videos', palette='crest')
plt.title("Top 10 Channels by Number of Videos Watched")
plt.xlabel("Videos Watched"); plt.ylabel("")
plt.savefig("C:/Users/ohdon/Downloads/bar_top10_channels.png", dpi=300, bbox_inches='tight')
plt.close()

# 6-3) 카테고리별 시청 시간 파이차트
plt.figure(figsize=(8,8))
plt.pie(category_watch['minutes'].head(5),
        labels=category_watch['category_id'].head(5),
        autopct='%1.1f%%', startangle=140)
plt.title("Share of Watch Time by Category (Top 5)")
plt.savefig("C:/Users/ohdon/Downloads/pie_category_watch.png", dpi=300, bbox_inches='tight')
plt.close()

# 6-4) 영상 길이 분포 히스토그램
plt.figure(figsize=(10,5))
sns.histplot(df['duration'], bins=50, color='skyblue')
plt.axvline(df['duration'].median(), color='red', linestyle='--', label='Median')
plt.title("Distribution of Video Durations (minutes)")
plt.xlabel("Minutes per Video"); plt.ylabel("Count")
plt.legend()
plt.savefig("C:/Users/ohdon/Downloads/hist_video_duration.png", dpi=300, bbox_inches='tight')
plt.close()

# 6-5) 세션 길이 vs 영상 수 스캐터플롯
plt.figure(figsize=(10,6))
sns.scatterplot(data=session_detail, x='videos', y='duration_minutes', 
                hue='is_binge', alpha=0.7)
plt.title("Session Duration vs Number of Videos")
plt.xlabel("Videos per Session"); plt.ylabel("Session Duration (minutes)")
plt.savefig("C:/Users/ohdon/Downloads/scatter_session_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 6-6) 일별 시청 시간 시계열
plt.figure(figsize=(15,6))
daily_stats['date'] = pd.to_datetime(daily_stats['date'])
plt.plot(daily_stats['date'], daily_stats['minutes'], alpha=0.7)
plt.title("Daily Watch Time Over Time")
plt.xlabel("Date"); plt.ylabel("Minutes Watched")
plt.xticks(rotation=45)
plt.savefig("C:/Users/ohdon/Downloads/timeseries_daily_watch.png", dpi=300, bbox_inches='tight')
plt.close()

print("🎨 모든 시각화 완료!")

# ---------- 7. 최종 요약 출력 ----------
print(f"""
📊 YouTube 시청 분석 완료!

데이터 요약:
- 총 영상 수: {len(df):,}개
- 총 시청 시간: {df['duration'].sum():.1f}분 ({df['duration'].sum()/60:.1f}시간)
- 분석 기간: {df['watched_at'].min().date()} ~ {df['watched_at'].max().date()}
- 총 세션 수: {df['session_id'].nunique():,}개
- 평균 세션당 영상 수: {df['session_videos'].mean():.1f}개
- 몰아보기 세션 비율: {(df['is_binge_session'].sum() / len(df) * 100):.1f}%

생성된 파일:
📄 CSV: 7개 집계 테이블
🎨 PNG: 6개 시각화 차트
💾 완전한 데이터: ytEnrichData_complete.csv
""")
