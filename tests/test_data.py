"""Tests pour le module churnguard.data."""

# On importe pandas pour pouvoir vérifier les types retournés
import pandas as pd

# On importe les deux fonctions qu'on veut tester
from churnguard.data import load_data, preprocess


def test_load_data_returns_dataframe(tmp_path):
    # tmp_path est un dossier temporaire créé automatiquement par pytest
    # Il est supprimé après le test, donc pas de fichier parasite sur le disque

    # On crée un fichier CSV minimal dans ce dossier temporaire
    csv_file = tmp_path / "test.csv"

    # On écrit une ligne d'en-tête avec les 21 colonnes du vrai dataset
    # puis une ligne de données fictives mais valides
    csv_file.write_text(
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "1234,Male,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,No,No,"
        "Month-to-month,Yes,Electronic check,70.0,840.0,No\n"
    )

    # On appelle la fonction qu'on teste en lui passant le chemin du CSV
    result = load_data(str(csv_file))

    # On vérifie que le résultat est bien un DataFrame pandas (et pas None ou autre)
    assert isinstance(result, pd.DataFrame)


def test_load_data_has_expected_columns(tmp_path):
    # Même principe : dossier temporaire + CSV minimal
    csv_file = tmp_path / "test.csv"

    # Même contenu que le test précédent (21 colonnes exactement)
    csv_file.write_text(
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "1234,Male,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,No,No,"
        "Month-to-month,Yes,Electronic check,70.0,840.0,No\n"
    )

    # On charge le CSV avec notre fonction
    result = load_data(str(csv_file))

    # result.columns contient la liste des noms de colonnes
    # len(...) compte combien il y en a — on attend exactement 21
    assert len(result.columns) == 21


def test_preprocess_returns_features_and_target():
    # On fabrique un DataFrame directement en Python (sans passer par un CSV)
    # C'est plus rapide et on contrôle exactement ce qu'il contient
    df = pd.DataFrame({
        # Identifiant client — sera supprimé par preprocess
        "customerID": ["1234"],
        # Variable catégorielle texte
        "gender": ["Male"],
        # Variable numérique entière (0 ou 1)
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        # Nombre de mois d'abonnement
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        # Montant mensuel en float
        "MonthlyCharges": [70.0],
        # TotalCharges est une chaîne dans le vrai CSV (d'où les guillemets)
        "TotalCharges": ["840.0"],
        # La cible : "Yes" = a résilié, "No" = n'a pas résilié
        "Churn": ["No"],
    })

    # On appelle preprocess et on décompacte le tuple retourné en X et y
    X, y = preprocess(df)

    # X doit être un DataFrame (les features = colonnes d'entrée du modèle)
    assert isinstance(X, pd.DataFrame)

    # y doit être une Series (la cible = ce qu'on veut prédire)
    assert isinstance(y, pd.Series)


def test_preprocess_handles_missing_total_charges():
    # Ce test vérifie le cas problématique du vrai dataset :
    # certaines lignes ont un espace " " dans TotalCharges au lieu d'un nombre
    # preprocess doit convertir ces espaces en NaN puis supprimer ces lignes

    df = pd.DataFrame({
        "customerID": ["1234", "5678"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 0],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "No"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No"],
        "OnlineBackup": ["No", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes"],
        "PaymentMethod": ["Electronic check", "Electronic check"],
        "MonthlyCharges": [70.0, 20.0],
        # Ligne 1 : valeur normale / Ligne 2 : espace vide = valeur manquante
        "TotalCharges": ["840.0", " "],
        "Churn": ["No", "Yes"],
    })

    # On appelle preprocess sur ce DataFrame avec une ligne problématique
    X, y = preprocess(df)

    # Après le dropna, il ne doit rester qu'une seule ligne (la ligne 1)
    # La ligne avec " " dans TotalCharges doit avoir été supprimée
    assert len(X) == 1
