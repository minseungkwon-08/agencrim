# ==============================
# CSV 파일 구조 확인 스크립트
# ==============================

import os
import pandas as pd

def check_csv_structure():
    data_dir = "/Users/minseung/Desktop/agencrim/data/rawdata"
    
    if not os.path.exists(data_dir):
        print(f"[오류] 폴더가 없습니다: {data_dir}")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    print("\n" + "="*80)
    print("CSV 파일 구조 분석")
    print("="*80 + "\n")
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        print(f"📄 파일: {filename}")
        print("-" * 80)
        
        # 여러 인코딩 시도
        encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
        df = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc, nrows=5)
                print(f"✓ 인코딩: {enc}")
                break
            except:
                continue
        
        if df is None:
            print("✗ 파일을 읽을 수 없습니다.\n")
            continue
        
        # 기본 정보
        print(f"Shape: {df.shape}")
        print(f"\n컬럼명 (총 {len(df.columns)}개):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. '{col}'")
        
        print(f"\n첫 3행 데이터:")
        print(df.head(3).to_string())
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    check_csv_structure()
