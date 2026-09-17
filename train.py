import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import dagshub

# TODO: sustituye por tu usuario y el nombre real del repo que crees en Dagshub
dagshub.init(repo_owner='victoriaguilllen', repo_name='wine-mlops-practica', mlflow=True)

# Cargar el conjunto de datos versionado con DVC
data = pd.read_csv('data/wine_dataset.csv')
X = data.drop(columns=['target'])
y = data['target']

# Iniciar un experimento de MLflow
with mlflow.start_run():
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Parámetro que iremos variando entre ejecuciones
    n_estimators = 100

    # Inicializar y entrenar el modelo
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Realizar predicciones y calcular la precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar el modelo entrenado en un archivo .pkl
    joblib.dump(model, 'model.pkl')

    # Registrar el modelo con MLflow
    mlflow.sklearn.log_model(
        model,
        "random-forest-wine-model",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )

    # Registrar parámetros y métricas
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")
