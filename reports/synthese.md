# Évaluation exécutée

7043 clients ; churn observé : 26.5%.
Séparation stratifiée 60/20/20 : 4225/1409/1409.
Modèle choisi sur validation : **logistique**, seuil **0.07**.
Sur test : précision 0.383, rappel 0.971, F1 0.549, average precision 0.629.
Lift du premier quintile sur test : 2.50.

Les coûts FP=10 et FN=100 sont des hypothèses illustratives, pas des coûts mesurés.
Le coût de classement ne modélise ni l'efficacité d'une remise ni le gain causal d'une campagne.
`targeting_test.csv` est une simulation rétrospective sur clients déjà étiquetés, pas une liste de vrais clients à contacter.
L'importance par permutation est calculée sur validation ; elle mesure une dépendance prédictive, pas une cause du départ.
L'historique ne contient pas de dates de photographie permettant de valider une anticipation temporelle.
Avant usage réel : définir une date de score, un horizon, vérifier la disponibilité des variables, contrôler les écarts par groupe et tester une campagne randomisée.
