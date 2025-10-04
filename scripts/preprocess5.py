import pandas as pd
import numpy as np
import os

# ==================== 설정 ====================
data_dir = "/Users/minseung/Desktop/agencrim/data/rawdata"
output_dir = "/Users/minseung/Desktop/agencrim/data/processed"
os.makedirs(output_dir, exist_ok=True)

# 지역명 표준화 사전
REGION_MAP = {
    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
    "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
    "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
    "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"
}

def standardize_region(text):
    """지역명을 표준화"""
    if pd.isna(text):
        return None
    text = str(text).strip()
    # 괄호 제거
    text = text.split('(')[0].strip()
    # 영문 제거
    text = text.split()[0] if ' ' in text else text
    # 매핑
    return REGION_MAP.get(text, text)

# ==================== 1. 독거노인수 ====================
print("\n1️⃣ 독거노인수 처리 중...")
df1 = pd.read_csv(f"{data_dir}/2023 시도별 독거노인수.csv", encoding="cp949")
# 2023년만 선택
df1 = df1[df1['연도'] == 2023].copy()
# 지역명 표준화
df1['region'] = df1['시도'].apply(standardize_region)
# 필요 컬럼만
df1 = df1[['region', '65-69세', '70-74세', '75-79세', '80-84세', '85세이상']]
print(f"   ✓ {df1.shape[0]}개 지역")

# ==================== 2. 디지털배움터 ====================
print("\n2️⃣ 디지털배움터 처리 중...")
df2 = pd.read_csv(f"{data_dir}/지역별 디지털배움터.csv", encoding="cp949")
# 광역지자체 표준화
df2['region'] = df2['광역지자체'].apply(standardize_region)
# 광역별 합계
df2 = df2.groupby('region', as_index=False)['교육인원'].sum()
df2 = df2.rename(columns={'교육인원': 'digital_education'})
print(f"   ✓ {df2.shape[0]}개 지역")

# ==================== 3. 인구 ====================
print("\n3️⃣ 인구 처리 중...")
df3 = pd.read_csv(f"{data_dir}/257인구.csv", encoding="cp949")
# 지역명 표준화
df3['region'] = df3['행정구역'].apply(standardize_region)
# "전국" 제거
df3 = df3[df3['region'] != "전국"]
# 쉼표 제거 및 숫자 변환
for col in df3.columns:
    if '2025년07월' in col:
        df3[col] = df3[col].str.replace(',', '').astype(float)
# 필요 컬럼만
pop_cols = ['region', '2025년07월_전체', '2025년07월_65세이상전체']
df3 = df3[pop_cols]
df3 = df3.rename(columns={
    '2025년07월_전체': 'population_total',
    '2025년07월_65세이상전체': 'population_elderly'
})
print(f"   ✓ {df3.shape[0]}개 지역")

# ==================== 4. 평생교육기관 ====================
print("\n4️⃣ 평생교육기관 처리 중...")
df4 = pd.read_csv(f"{data_dir}/지역별 평생교육기관.csv", encoding="utf-8-sig")
# Unnamed: 1에 실제 지역명이 있음
df4['region'] = df4['Unnamed: 1'].apply(standardize_region)
# NaN이 아닌 것만
df4 = df4[df4['region'].notna()]
# 표준 지역명만 (17개만)
df4 = df4[df4['region'].isin(REGION_MAP.values())]
# 숫자 변환
for col in ['기관수', '프로그램수', '학습자수']:
    df4[col] = df4[col].str.replace(',', '').astype(float)
# 필요 컬럼만
df4 = df4[['region', '기관수', '프로그램수', '학습자수']]
df4 = df4.rename(columns={
    '기관수': 'education_institutions',
    '프로그램수': 'education_programs',
    '학습자수': 'education_students'
})
print(f"   ✓ {df4.shape[0]}개 지역")

# ==================== 5. 위도경도 ====================
print("\n5️⃣ 위도경도 처리 중...")
df5 = pd.read_csv(f"{data_dir}/위도경도.csv", encoding="utf-8-sig")
# do 컬럼을 표준화
df5['region'] = df5['do'].apply(standardize_region)
# 광역별 평균 좌표
df5 = df5.groupby('region', as_index=False)[['latitude', 'longitude']].mean()
print(f"   ✓ {df5.shape[0]}개 지역")

# ==================== 병합 ====================
print("\n🔗 데이터 병합 중...")
# 순차적으로 병합 (outer join으로 모든 지역 포함)
master = df1
master = pd.merge(master, df2, on='region', how='outer')
master = pd.merge(master, df3, on='region', how='outer')
master = pd.merge(master, df4, on='region', how='outer')
master = pd.merge(master, df5, on='region', how='outer')

# 정렬
master = master.sort_values('region').reset_index(drop=True)

print(f"\n✅ 완료!")
print(f"   최종 shape: {master.shape}")
print(f"   지역 수: {master.shape[0]}")
print(f"\n지역 목록:")
for i, region in enumerate(master['region'], 1):
    print(f"   {i:2d}. {region}")

# ==================== 저장 ====================
output_path = f"{output_dir}/cleaned_final.csv"
master.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n💾 저장: {output_path}")

# 샘플 출력
print(f"\n📊 데이터 샘플 (첫 3행):")
print(master.head(3).to_string())
