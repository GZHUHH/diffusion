import argparse
from .core import read_table, load_bundle, predict

def main():
    p = argparse.ArgumentParser(description='Predict log10(D) and D in cm²/s')
    p.add_argument('input'); p.add_argument('output'); p.add_argument('--model')
    a = p.parse_args()
    result = predict(read_table(a.input), load_bundle(a.model))
    if a.output.lower().endswith('.csv'):
        result.to_csv(a.output,index=False,encoding='utf-8-sig')
    else:
        result.to_excel(a.output,index=False)
    print(f'Saved {len(result)} predictions: {a.output}')

if __name__ == '__main__': main()
