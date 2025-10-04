# ==============================
# 전처리 스크립트: preprocess.py (최종 수정본)
# ==============================

import os
import pandas as pd
import numpy as np
import re

# ------------------------------
# 1. 데이터 불러오기
# ------------------------------
def load_csv_with_encoding(filepath):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df, enc
        except:
            continue
    return None, None


# ------------------------------
# 2. 숫자 문자열 변환
# ------------------------------
def convert_numeric_strings(df):
    """쉼표 포함 숫자를 실제 숫자로 변환"""
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                cleaned = df[col].astype(str).str.replace(',', '', regex=False)
                numeric = pd.to_numeric(cleaned, errors='coerce')
                # 변환 성공한 경우만 적용
                if not numeric.isna().all():
                    df[col] = numeric.fillna(df[col])
            except:
                pass
    return df


# ------------------------------
# 3. 지역명 표준화
# ------------------------------
def standardize_region_names(series):
    """지역명 Series를 표준화"""
    # 문자열 변환 및 정리
    series = series.fillna("").astype(str).str.strip()
    
    # 빈 문자열 제거
    mask = (series.str.len() > 0) & (series.str.lower() != "nan")
    
    # 괄호 제거
    series = series.str.replace(r'\s*\([^)]*\)', '', regex=True)
    # 영문 제거
    series = series.str.replace(r'\s+[A-Za-z]+', '', regex=True)
    # 공백 정리
    series = series.str.strip()
    
    # 표준 명칭 매핑
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
    
    series = series.replace(region_map)
    
    return series, mask


# ------------------------------
# 4. 파일별 전처리
# ------------------------------
def preprocess_file(df, filename):
    """파일 종류에 따라 적절한 전처리"""
    
    # 1. 평생교육기관 파일
    if "평생교육" in filename:
        # Unnamed: 1이 실제 지역명
        if "Unnamed: 1" in df.columns:
            df["region"] = df["Unnamed: 1"]
        elif "구분" in df.columns:
            df["region"] = df["구분"]
        
        # 불필요한 컬럼 제거
        df = df.drop(columns=["구분", "Unnamed: 1"], errors='ignore')
        
        # 숫자 변환
        df = convert_numeric_strings(df)
        
        # NaN 제거 (첫 줄 헤더 등)
        df = df[df["region"].notna()]
        
        # 지역명 표준화
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        # 권역 데이터 제외 + "시도" 같은 헤더 제외
        exclude = ["합계", "전국", "수도권", "비수도권", "권역", "시도"]
        df = df[~df["region"].str.lower().isin(exclude)]
        
        return df
    
    # 2. 위도경도 파일
    elif "위도경도" in filename:
        # do 컬럼을 region으로 사용
        if "do" in df.columns:
            df["region"] = df["do"]
        elif "docity" in df.columns:
            # docity에서 광역시도만 추출 (예: "강원강릉시" -> "강원")
            df["region"] = df["docity"].astype(str).str[:2]
        
        # 지역명 표준화
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        # 시/군별 좌표를 광역 단위로 평균 계산
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.groupby("region", as_index=False).agg({
                "latitude": "mean",
                "longitude": "mean"
            })
            print(f"  ℹ️  시/군 좌표 → 광역 평균 계산")
        
        return df
    
    # 3. 디지털배움터 파일
    elif "디지털배움터" in filename:
        if "광역지자체" in df.columns:
            df["region"] = df["광역지자체"]
        
        # 숫자 변환
        df = convert_numeric_strings(df)
        
        # 지역명 표준화
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        # 기초지자체별 데이터를 광역으로 집계
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = df.groupby("region", as_index=False)[numeric_cols].sum()
        
        return df
    
    # 4. 독거노인수 파일
    elif "독거노인" in filename or "시도별" in filename:
        if "시도" in df.columns:
            df["region"] = df["시도"]
            df = df.drop(columns=["시도"], errors='ignore')
        
        # 숫자 변환
        df = convert_numeric_strings(df)
        
        # 지역명 표준화
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        # 연도별 중복 제거 (최신 데이터)
        if "연도" in df.columns:
            df = df.sort_values("연도", ascending=False).drop_duplicates("region", keep="first")
        
        return df
    
    # 5. 인구 파일
    elif "인구" in filename:
        if "행정구역" in df.columns:
            df["region"] = df["행정구역"]
            df = df.drop(columns=["행정구역"], errors='ignore')
        
        # 숫자 변환
        df = convert_numeric_strings(df)
        
        # 지역명 표준화
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        return df
    
    # 6. 기타 파일
    else:
        # region 컬럼 찾기
        region_keywords = ["행정구역", "시도", "지역", "광역지자체", "구분"]
        for col in df.columns:
            if any(kw in col for kw in region_keywords):
                df["region"] = df[col]
                break
        
        if "region" not in df.columns:
            return None
        
        df = convert_numeric_strings(df)
        df["region"], mask = standardize_region_names(df["region"])
        df = df[mask]
        
        return df


# ------------------------------
# 5. 결측치 및 이상치 처리
# ------------------------------
def clean_data(df):
    """결측치 및 이상치 처리"""
    # 결측치
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
    
    # 이상치 (z-score > 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        values = df[col].astype(float)
        mean = values.mean()
        std = values.std()
        
        if std > 0:
            z_scores = (values - mean) / std
            outliers = np.abs(z_scores) > 3
            if outliers.sum() > 0:
                median = values.median()
                df[col] = values.where(~outliers, median)
    
    return df


# ------------------------------
# 6. 메인 실행
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
    print(f"데이터 전처리 시작: {len(files)}개 파일")
    print(f"{'='*60}\n")

    all_dfs = {}
    
    for filename in files:
        print(f"📁 {filename}")
        filepath = os.path.join(data_dir, filename)
        
        # 불러오기
        df, enc = load_csv_with_encoding(filepath)
        if df is None:
            print(f"  ✗ 불러오기 실패\n")
            continue
        
        print(f"  ✓ 불러오기 성공 ({enc})")
        print(f"  원본: {df.shape}, 컬럼: {list(df.columns)[:3]}...")
        
        # 전처리
        df_processed = preprocess_file(df, filename)
        
        if df_processed is None or "region" not in df_processed.columns:
            print(f"  ✗ region 컬럼 없음 - 제외\n")
            continue
        
        # 데이터 정리
        df_processed = clean_data(df_processed)
        
        print(f"  처리 후: {df_processed.shape}")
        print(f"  지역: {sorted(df_processed['region'].unique())[:3]}...")
        print(f"  ✓ 완료\n")
        
        all_dfs[filename] = df_processed

    # 병합
    print(f"{'='*60}")
    print(f"병합 시작: {len(all_dfs)}개 파일")
    print(f"{'='*60}\n")
    
    if len(all_dfs) == 0:
        print("[오류] 처리된 파일이 없습니다")
        return
    
    # 첫 번째 파일을 기준으로 시작
    file_list = list(all_dfs.items())
    master_df = file_list[0][1]
    print(f"기준: {file_list[0][0]} ({master_df.shape})")
    
    # 나머지 파일들과 병합
    for filename, df in file_list[1:]:
        before = len(master_df)
        master_df = pd.merge(master_df, df, on="region", how="outer")
        print(f"+ {filename}: {before}행 → {len(master_df)}행")
    
    print(f"\n{'='*60}")
    print(f"✅ 병합 완료!")
    print(f"{'='*60}\n")
    
    # 저장
    output_path = os.path.join(output_dir, "cleaned_master.csv")
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"💾 저장: {output_path}")
    print(f"   Shape: {master_df.shape}")
    print(f"   지역 수: {master_df['region'].nunique()}")
    print(f"\n최종 지역 목록:")
    for i, region in enumerate(sorted(master_df['region'].unique()), 1):
        print(f"   {i:2d}. {region}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
