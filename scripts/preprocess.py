# ==============================
# 전처리 스크립트: preprocess.py (지역명 표준화 강화)
# ==============================

import os
import pandas as pd
import numpy as np
import re

# ------------------------------
# 1. 데이터 불러오기
# ------------------------------
def load_csv_with_encoding(filepath):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  ✓ 불러오기 성공 (encoding={enc})")
            return df, enc
        except:
            continue
    print(f"  ✗ 모든 인코딩 실패")
    return None, None


# ------------------------------
# 2. 컬럼명 정리
# ------------------------------
def clean_column_names(df):
    new_columns = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        col_clean = col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("·", "_")
        
        # 지역 관련
        region_keywords = ["행정구역", "행정", "시도", "지역", "광역지자체", "구분", "region", "Çà"]
        if any(keyword in col for keyword in region_keywords):
            new_columns[col] = "region"
        # 기타
        elif "연령" in col or "age" in col_lower:
            new_columns[col] = "age"
        elif "노인" in col or "elderly" in col_lower:
            new_columns[col] = "elderly"
        elif "교육인원" in col or "education" in col_lower:
            new_columns[col] = "education_people"
        elif "기초지자체" in col:
            new_columns[col] = "city"
        elif "기관" in col or "institutions" in col_lower:
            new_columns[col] = "institutions"
        elif "위도" in col or "latitude" in col_lower:
            new_columns[col] = "latitude"
        elif "경도" in col or "longitude" in col_lower:
            new_columns[col] = "longitude"
        else:
            new_columns[col] = col_clean.lower()
    
    df = df.rename(columns=new_columns)
    return df


# ------------------------------
# 3. 지역명 표준화 (핵심 개선!)
# ------------------------------
def standardize_region_names(df):
    """
    지역명을 표준화합니다.
    - 영문 제거 (서울 Seoul -> 서울특별시)
    - 괄호와 코드 제거 (서울특별시 (1100000000) -> 서울특별시)
    - 약칭을 정식 명칭으로 변환 (서울 -> 서울특별시)
    """
    if "region" not in df.columns:
        return df
    
    # 문자열로 변환
    df["region"] = df["region"].astype(str).str.strip()
    
    # 1. 괄호와 내용 제거: "서울특별시  (1100000000)" -> "서울특별시"
    df["region"] = df["region"].str.replace(r'\s*\([^)]*\)', '', regex=True)
    
    # 2. 영문 제거: "서울 Seoul" -> "서울"
    df["region"] = df["region"].str.replace(r'\s+[A-Za-z]+', '', regex=True)
    
    # 3. 공백 정리
    df["region"] = df["region"].str.strip()
    
    # 4. 표준 명칭 매핑
    region_map = {
        "서울": "서울특별시",
        "부산": "부산광역시",
        "대구": "대구광역시",
        "인천": "인천광역시",
        "광주": "광주광역시",
        "대전": "대전광역시",
        "울산": "울산광역시",
        "세종": "세종특별자치시",
        "경기": "경기도",
        "강원": "강원특별자치도",
        "강원도": "강원특별자치도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전북특별자치도",
        "전라북도": "전북특별자치도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주특별자치도",
    }
    
    df["region"] = df["region"].replace(region_map)
    
    # 5. 권역 데이터 제거 (합계, 수도권, 비수도권 등)
    exclude_keywords = ["합계", "전국", "수도권", "비수도권", "권역", "NaN", "nan"]
    df = df[~df["region"].isin(exclude_keywords)]
    
    return df


# ------------------------------
# 4. 광역지자체 데이터 집계
# ------------------------------
def aggregate_by_region(df):
    """
    기초지자체 데이터를 광역지자체 단위로 집계합니다.
    (디지털배움터 파일용)
    """
    if "city" in df.columns and "region" in df.columns:
        # 숫자형 컬럼만 합계
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_agg = df.groupby("region", as_index=False)[numeric_cols].sum()
            return df_agg
    return df


# ------------------------------
# 5. 결측치 처리
# ------------------------------
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")
    return df


# ------------------------------
# 6. 이상치 처리
# ------------------------------
def handle_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_values = df[col].astype(float)
        mean_val = col_values.mean()
        std_val = col_values.std()
        
        if std_val == 0:
            continue
        
        z_scores = (col_values - mean_val) / std_val
        outlier_mask = np.abs(z_scores) > 3
        
        if outlier_mask.sum() > 0:
            median_val = col_values.median()
            df[col] = col_values.where(~outlier_mask, median_val)
    
    return df


# ------------------------------
# 7. 숫자 문자열을 실제 숫자로 변환
# ------------------------------
def convert_numeric_strings(df):
    """
    '51,159,889' 같은 쉼표 포함 문자열을 숫자로 변환
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # 문자열 컬럼만
            # 쉼표 제거 후 숫자 변환 시도
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
            except:
                pass  # 변환 실패하면 그대로 유지
    return df


# ------------------------------
# 8. 메인 실행
# ------------------------------
def main():
    data_dir = "/Users/minseung/Desktop/agencrim/data/rawdata"
    output_dir = "/Users/minseung/Desktop/agencrim/data/processed"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"[오류] 데이터 폴더가 존재하지 않습니다: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"\n{'='*60}")
    print(f"[데이터 폴더 확인] {len(files)}개 CSV 파일 발견")
    print(f"{'='*60}\n")

    if not files:
        print("[오류] CSV 파일이 없습니다.")
        return

    all_dfs = []
    file_info = []
    
    for filename in files:
        print(f"📁 처리 중: {filename}")
        filepath = os.path.join(data_dir, filename)
        
        df, encoding = load_csv_with_encoding(filepath)
        
        if df is None:
            print(f"  ⚠️  건너뜀\n")
            continue
        
        print(f"  원본 컬럼: {list(df.columns)[:3]}...")
        print(f"  원본 shape: {df.shape}")
        
        # 전처리
        df = clean_column_names(df)
        
        if "region" not in df.columns:
            print(f"  ✗ 'region' 컬럼 없음 → 제외\n")
            continue
        
        print(f"  ✓ 'region' 컬럼 발견")
        
        # 숫자 변환
        df = convert_numeric_strings(df)
        
        # 지역명 표준화
        df = standardize_region_names(df)
        
        # 기초지자체 데이터 집계
        if "city" in df.columns:
            print(f"  ℹ️  기초지자체 데이터 → 광역 단위로 집계")
            df = aggregate_by_region(df)
        
        df = handle_missing_values(df)
        df = handle_outliers(df)
        
        # region 기준으로 중복 제거 (최신 데이터 유지)
        if "연도" in df.columns:
            df = df.sort_values("연도", ascending=False).drop_duplicates("region", keep="first")
        else:
            df = df.drop_duplicates("region", keep="first")
        
        print(f"  표준화 후 지역: {sorted(df['region'].unique())[:5]}...")
        print(f"  ✓ 전처리 완료 (shape: {df.shape})\n")
        
        all_dfs.append(df)
        file_info.append(filename)

    # 병합
    print(f"{'='*60}")
    if len(all_dfs) == 0:
        print("[오류] 처리된 파일이 없습니다.")
        return
    elif len(all_dfs) == 1:
        print(f"[정보] 파일 1개만 처리됨")
        master_df = all_dfs[0]
    else:
        print(f"[병합 시작] {len(all_dfs)}개 파일 병합 중...")
        master_df = all_dfs[0]
        print(f"  기준: {file_info[0]}")
        print(f"    - shape: {master_df.shape}")
        print(f"    - 지역 수: {master_df['region'].nunique()}")
        
        for i, other_df in enumerate(all_dfs[1:], 1):
            before_rows = len(master_df)
            master_df = pd.merge(master_df, other_df, on="region", how="outer")
            print(f"  + {file_info[i]}")
            print(f"    - 병합 전: {before_rows}행 → 병합 후: {len(master_df)}행")
        
        print(f"  ✓ 병합 완료!")

    # 저장
    print(f"{'='*60}")
    output_path = os.path.join(output_dir, "cleaned_master.csv")
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ [저장 완료] {output_path}")
    print(f"   최종 shape: {master_df.shape}")
    print(f"   지역 목록 ({master_df['region'].nunique()}개):")
    for region in sorted(master_df['region'].unique()):
        print(f"     - {region}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
