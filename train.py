"""Comparaison reproductible : référence, logistique et arbre. Python 3.12."""
from pathlib import Path
from urllib.request import urlopen
import argparse, json, hashlib, shutil, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay)

ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
SEED=42

def prepare(frame):
    frame=frame.copy()
    required={'customerID','Churn','tenure','MonthlyCharges','TotalCharges'}
    if not required.issubset(frame): raise ValueError(f'Colonnes manquantes : {required-set(frame)}')
    if frame.customerID.isna().any() or frame.customerID.duplicated().any():
        raise ValueError('Identifiants clients manquants ou dupliqués')
    if not frame.Churn.isin(['Yes','No']).all(): raise ValueError('Cible invalide')
    frame['TotalCharges']=pd.to_numeric(frame.TotalCharges,errors='coerce')
    numeric=['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']
    # gender reste disponible dans les données mais est exclu du ciblage.
    x=frame.drop(columns=['customerID','Churn','gender'])
    categorical=[c for c in x if c not in numeric]
    pre=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                              ('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])
    return frame,x,frame.Churn.eq('Yes').astype(int),pre

def metrics(y,p,threshold):
    pred=(np.asarray(p)>=threshold).astype(int)
    return {'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'average_precision':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run(fp_cost=10.0,fn_cost=100.0):
    if fp_cost<=0 or fn_cost<=0: raise ValueError('Les coûts doivent être positifs.')
    reports=ROOT/'reports'; models=ROOT/'models'
    reports.mkdir(exist_ok=True); models.mkdir(exist_ok=True)
    raw=ROOT/'data/raw/telco.csv'; raw.parent.mkdir(parents=True,exist_ok=True)
    if not raw.exists():
        with urlopen(URL,timeout=60) as r,raw.with_suffix('.part').open('wb') as f: shutil.copyfileobj(r,f)
        raw.with_suffix('.part').replace(raw)
    df,x,y,pre=prepare(pd.read_csv(raw))
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=SEED)
    train_idx,val_idx=train_test_split(train_idx,test_size=.25,stratify=y.iloc[train_idx],random_state=SEED)
    assert not (set(train_idx)&set(test_idx) or set(train_idx)&set(val_idx) or set(val_idx)&set(test_idx))
    xt,xv,xe=x.iloc[train_idx],x.iloc[val_idx],x.iloc[test_idx]
    yt,yv,ye=y.iloc[train_idx],y.iloc[val_idx],y.iloc[test_idx]
    candidates={
      'reference':(DummyClassifier(strategy='prior'),{}),
      'logistique':(LogisticRegression(max_iter=1500,random_state=SEED),{'model__C':[.1,1,10]}),
      'arbre':(DecisionTreeClassifier(random_state=SEED),{'model__max_depth':[3,5,8], 'model__min_samples_leaf':[20,50]}),
    }
    fitted={}; comparison=[]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for name,(estimator,grid) in candidates.items():
        start=time.perf_counter()
        search=GridSearchCV(Pipeline([('preprocess',pre),('model',estimator)]),grid,
                            scoring='average_precision',cv=cv,n_jobs=1,error_score='raise')
        search.fit(xt,yt); fitted[name]=search.best_estimator_
        p=search.predict_proba(xv)[:,1]
        comparison.append({'model':name,'cv_average_precision':search.best_score_,
                           'validation_average_precision':average_precision_score(yv,p),
                           'params':search.best_params_, 'fit_seconds':time.perf_counter()-start})
    # Choix du modèle et seuil exclusivement sur validation, avant consultation test.
    winner=max(comparison,key=lambda r:r['validation_average_precision'])['model']
    model=fitted[winner]; pv=model.predict_proba(xv)[:,1]
    thresholds=np.r_[np.linspace(0,1,101),1.000001]
    costs=[]
    for threshold in thresholds:
        tn,fp,fn,tp=confusion_matrix(yv,pv>=threshold,labels=[0,1]).ravel()
        costs.append({'threshold':float(threshold),'fp':int(fp),'fn':int(fn),
                      'cost':float(fp*fp_cost+fn*fn_cost),'contact_rate':float((pv>=threshold).mean())})
    best=min(costs,key=lambda r:(r['cost'],-r['threshold']))
    threshold=best['threshold']; test={}
    for name,m in fitted.items(): test[name]=metrics(ye,m.predict_proba(xe)[:,1],.5)
    pe=model.predict_proba(xe)[:,1]
    selected=metrics(ye,pe,threshold)
    confusion=ConfusionMatrixDisplay.from_predictions(ye,pe>=threshold,display_labels=['Reste','Part'],cmap='Blues',colorbar=False)
    confusion.ax_.set_title(f'{winner} · test · seuil {threshold:.2f}')
    plt.tight_layout(); plt.savefig(reports/'confusion.png',dpi=150); plt.close()
    fig,ax=plt.subplots(figsize=(7,5))
    for name,m in fitted.items(): PrecisionRecallDisplay.from_predictions(ye,m.predict_proba(xe)[:,1],name=name,ax=ax)
    ax.set_title('Comparaison sur test — seuils non réglés sur test')
    fig.tight_layout(); fig.savefig(reports/'precision_recall.png',dpi=150); plt.close(fig)
    importance=permutation_importance(model,xv,yv,n_repeats=5,random_state=SEED,scoring='average_precision',n_jobs=1)
    pd.DataFrame({'feature':x.columns,'mean_ap_loss':importance.importances_mean,
                  'std':importance.importances_std}).sort_values('mean_ap_loss',ascending=False).to_csv(reports/'importance_validation.csv',index=False)
    target=pd.DataFrame({'customer_id':df.customerID.iloc[test_idx].to_numpy(),'observed_churn':ye.to_numpy(),
                         'score':pe,'contact_proposed':pe>=threshold}).sort_values('score',ascending=False)
    target.to_csv(reports/'targeting_test.csv',index=False)
    ranked=target.head(max(1,int(np.ceil(.2*len(target)))))
    lift=float(ranked.observed_churn.mean()/ye.mean())
    pd.DataFrame(costs).to_csv(reports/'threshold_validation.csv',index=False)
    pd.DataFrame({'customer_id':df.customerID,'partition':np.select([df.index.isin(train_idx),df.index.isin(val_idx)],['train','validation'],default='test')}).to_csv(reports/'partitions.csv',index=False)
    result={'seed':SEED,'rows':len(df),'missing_total_charges':int(df.TotalCharges.isna().sum()),
            'churn_rate':float(y.mean()),'split_sizes':{'train':len(yt),'validation':len(yv),'test':len(ye)},
            'source_sha256':hashlib.file_digest(raw.open('rb'),'sha256').hexdigest(),
            'comparison_validation':comparison,'test_at_0_5':test,'winner':winner,
            'threshold':threshold,'cost_assumptions':{'false_positive':fp_cost,'false_negative':fn_cost},
            'selected_test':selected,'lift_top_20_percent_test':lift}
    (reports/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    joblib.dump({'pipeline':model,'threshold':threshold,'features':list(x.columns)},models/'churn.joblib')
    restored=joblib.load(models/'churn.joblib')
    np.testing.assert_allclose(restored['pipeline'].predict_proba(xe)[:,1],pe)
    (reports/'synthese.md').write_text(f'''# Évaluation exécutée

{len(df)} clients ; churn observé : {y.mean():.1%}.
Séparation stratifiée 60/20/20 : {len(yt)}/{len(yv)}/{len(ye)}.
Modèle choisi sur validation : **{winner}**, seuil **{threshold:.2f}**.
Sur test : précision {selected['precision']:.3f}, rappel {selected['recall']:.3f}, F1 {selected['f1']:.3f}, average precision {selected['average_precision']:.3f}.
Lift du premier quintile sur test : {lift:.2f}.

Les coûts FP={fp_cost:g} et FN={fn_cost:g} sont des hypothèses illustratives, pas des coûts mesurés.
Le coût de classement ne modélise ni l'efficacité d'une remise ni le gain causal d'une campagne.
`targeting_test.csv` est une simulation rétrospective sur clients déjà étiquetés, pas une liste de vrais clients à contacter.
L'importance par permutation est calculée sur validation ; elle mesure une dépendance prédictive, pas une cause du départ.
L'historique ne contient pas de dates de photographie permettant de valider une anticipation temporelle.
Avant usage réel : définir une date de score, un horizon, vérifier la disponibilité des variables, contrôler les écarts par groupe et tester une campagne randomisée.
''',encoding='utf-8')
    print(json.dumps({'winner':winner,'threshold':threshold,'test':selected},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--fp-cost',type=float,default=10); parser.add_argument('--fn-cost',type=float,default=100)
    args=parser.parse_args(); run(args.fp_cost,args.fn_cost)
