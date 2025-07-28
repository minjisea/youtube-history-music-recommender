from bs4 import BeautifulSoup

# HTML 파일 열기
# with open('C:/Users/ohdon/Downloads/시청 기록.html', 'r', encoding='utf-8') as f:
#     lines = [next(f) for _ in range(120)]  # 처음 20줄만 읽기

# for i, line in enumerate(lines, start=1):
#     print(f'Line {i}: {line.strip()}')

import pandas as pd
from datetime import datetime, timedelta

# 1. 파일 열기
with open('C:/Users/ohdon/Downloads/시청 기록.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

# 2. 시청 기록 블록 추출
records = soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')

# 3. 날짜 문자열 파싱 함수
def parse_korean_dot_datetime(text):
    text = text.replace('오전', 'AM').replace('오후', 'PM').replace('KST', '').strip()
    try:
        return datetime.strptime(text, "%Y. %m. %d. %p %I:%M:%S")
    except Exception as e:
        return None

# 4. 기준일: 최근 3년
three_years_ago = datetime.now() - timedelta(days=3*365)

# 5. 데이터 수집
data = []

for r in records:
    try:
        a_tag = r.find('a')
        if not a_tag:
            continue

        url = a_tag['href']
        if 'https://myaccount.google.com/activitycontrols' in url:
            continue  # 제외 조건 적용

        title = a_tag.text.strip()
        raw_text = r.get_text(separator="|", strip=True).split("|")
        timestamp_raw = raw_text[-1]
        timestamp = parse_korean_dot_datetime(timestamp_raw)

        if not timestamp or timestamp < three_years_ago:
            continue  # 3년 이내만 포함

        data.append({
            'title': title,
            'url': url,
            'timestamp': timestamp
        })

    except Exception as e:
        continue

# 6. DataFrame 변환
df = pd.DataFrame(data)

# 7. 결과 확인
print(df.head())
print(f"최근 3년 기록 수: {len(df)}")


output_path = 'C:/Users/ohdon/Downloads/시청기록_최근3년.csv'  # 저장 경로 수정 가능
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"CSV 파일 저장 완료: {output_path}")
