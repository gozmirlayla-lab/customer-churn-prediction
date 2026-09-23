"""Scorer un CSV de nouveaux clients avec un modèle local de confiance."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('--output',default='predictions.csv')
    args=p.parse_args()
    model=joblib.load(Path(__file__).parent/'models/churn.joblib')
    df=pd.read_csv(args.csv); missing=set(model['features'])-set(df)
    if missing: raise ValueError(f'Variables absentes : {missing}')
    df['TotalCharges']=pd.to_numeric(df.TotalCharges,errors='coerce')
    score=model['pipeline'].predict_proba(df[model['features']])[:,1]
    pd.DataFrame({'customer_id':df.get('customerID',df.index),'score':score,
                  'contact_proposed':score>=model['threshold']}).to_csv(args.output,index=False)

if __name__=='__main__': main()
