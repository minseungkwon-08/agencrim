import pandas as pd
import os

data_dir = "/Users/minseung/Desktop/agencrim/data/rawdata"

print("="*60)
print("각 파일의 구조를 확인합니다")
print("="*60 + "\n")

files = {
    "1. 독거노인수": "2023 시도별 독거노인수.csv",
    "2. 디지털배움터": "지역별 디지털배움터.csv",
    "3. 인구": "257인구.csv",
    "4. 평생교육": "지역별 평생교육기관.csv",
    "5. 위도경도": "위도경도.csv"
}

for name, filename in files.items():
    filepath = os.path.join(data_dir, filename)
    print(f"\n{name}: {filename}")
    print("-" * 60)
    
    # 여러 인코딩 시도
    for enc in ["utf-8-sig", "cp949"]:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"✓ 인코딩: {enc}")
            print(f"Shape: {df.shape} (행={df.shape[0]}, 열={df.shape[1]})")
            print(f"컬럼: {list(df.columns)}")
            print(f"\n첫 번째 컬럼의 고유값 수: {df[df.columns[0]].nunique()}")
            print(f"첫 번째 컬럼 샘플 (3개):")
            print(df[df.columns[0]].head(3).tolist())
            break
        except:
            continue

print("\n" + "="*60)
