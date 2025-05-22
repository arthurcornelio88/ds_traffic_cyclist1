
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


# Cleaning and transforming class

class RawCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keep_compteur=True):
        self.keep_compteur = keep_compteur

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("✅ RawCleanerTransformer called")
        X = X.copy()

        # D'abord nettoyer les noms pour éviter erreurs
        X.columns = (
            X.columns
            .str.replace(r'\W+', '_', regex=True)
            .str.replace(r'^_+', '', regex=True)
            .str.lower()
            .str.strip()
        )

        # Conversion de la colonne datetime
        X['date_et_heure_de_comptage'] = pd.to_datetime(
            X['date_et_heure_de_comptage'], utc=True).dt.tz_convert('Europe/Paris')

        # Feature engineering
        X['heure'] = X['date_et_heure_de_comptage'].dt.hour
        X['jour_mois'] = X['date_et_heure_de_comptage'].dt.day
        X['mois'] = X['date_et_heure_de_comptage'].dt.month
        X['annee'] = X['date_et_heure_de_comptage'].dt.year
        X['jour_semaine'] = X['date_et_heure_de_comptage'].dt.day_name()

        # Suppression explicite de la colonne timestamp
        if 'date_et_heure_de_comptage' in X.columns:
            X.drop(columns='date_et_heure_de_comptage', inplace=True)

        # Coordonnées
        X[['latitude', 'longitude']] = X['coordonnées_géographiques'].str.split(',', expand=True).astype(float)

        colonnes_a_supprimer = [
            'mois_annee_comptage', 'identifiant_du_site_de_comptage', 'identifiant_du_compteur',
            'nom_du_site_de_comptage', 'lien_vers_photo_du_site_de_comptage',
            'identifiant_technique_compteur', 'id_photos', "date_d_installation_du_site_de_comptage",
            'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage',
            'coordonnées_géographiques'
        ]
        X.drop(columns=[col for col in colonnes_a_supprimer if col in X.columns], inplace=True)

        # Types
        X['latitude'] = X['latitude'].astype('float32')
        X['longitude'] = X['longitude'].astype('float32')
        X['jour_semaine'] = X['jour_semaine'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }).astype('int8')


        if self.keep_compteur:
            X['nom_du_compteur'] = X['nom_du_compteur'].astype('category')
        else:
            X.drop(columns='nom_du_compteur', inplace=True)

        # 🔒 Ordre canonique des colonnes
        X = X[sorted(X.columns)]

        return X

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['heure_sin'] = np.sin(2 * np.pi * X['heure'] / 24)
        X['heure_cos'] = np.cos(2 * np.pi * X['heure'] / 24)
        X['mois_sin'] = np.sin(2 * np.pi * X['mois'] / 12)
        X['mois_cos'] = np.cos(2 * np.pi * X['mois'] / 12)
        X['annee'] = X['annee'].map({2024: 0, 2025: 1})
        return X.drop(columns=['heure', 'mois'])

## NN class

class NNPipeline:
    def __init__(self, embedding_dim=8):
        self.embedding_dim = embedding_dim
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.cleaner = Pipeline([
            ('raw', RawCleanerTransformer(keep_compteur=True)),
            ('feat', TimeFeatureTransformer())
        ])
        self.model = None
        self.n_features = None

    def build_model(self, n_compteurs, n_features):
        input_id = Input(shape=(1,), name='compteur_id')
        input_dense = Input(shape=(n_features,), name='features_scaled')

        x_id = Embedding(input_dim=n_compteurs, output_dim=self.embedding_dim)(input_id)
        x_id = Flatten()(x_id)

        x = Concatenate()([x_id, input_dense])
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=[input_id, input_dense], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, X, y, epochs=50, batch_size=128):
        X_clean = self.cleaner.fit_transform(X)
        X_id = self.label_encoder.fit_transform(X_clean['nom_du_compteur']).reshape(-1, 1)
        X_dense = X_clean.drop(columns='nom_du_compteur')
        X_scaled = self.scaler.fit_transform(X_dense)

        n_compteurs = int(X_id.max()) + 1
        self.n_features = X_scaled.shape[1]
        self.model = self.build_model(n_compteurs, self.n_features)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-3, restore_best_weights=True)

        self.model.fit(
            [X_id, X_scaled], y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

    def predict(self, X):
        X_clean = self.cleaner.transform(X)
        X_id = self.label_encoder.transform(X_clean['nom_du_compteur']).reshape(-1, 1)
        X_dense = X_clean.drop(columns='nom_du_compteur')
        X_scaled = self.scaler.transform(X_dense)
        return self.model.predict([X_id, X_scaled])

    def predict_clean(self, X_raw):
        X_id, X_scaled = self.preprocess(X_raw)
        return self.model.predict([X_id, X_scaled])

    def preprocess(self, X):
        X_clean = self.cleaner.transform(X)
        X_id = self.label_encoder.transform(X_clean['nom_du_compteur']).reshape(-1, 1)
        X_dense = X_clean.drop(columns='nom_du_compteur')
        X_scaled = self.scaler.transform(X_dense)
        return X_id, X_scaled

    def save(self, path_prefix='nn_prod'):
        joblib.dump(self.cleaner, f'{path_prefix}_cleaner.joblib')
        joblib.dump(self.label_encoder, f'{path_prefix}_label_encoder.joblib')
        joblib.dump(self.scaler, f'{path_prefix}_scaler.joblib')
        self.model.save(f'{path_prefix}_model.keras')

    @classmethod
    def load(cls, path_prefix='nn_prod'):
        instance = cls()
        instance.cleaner = joblib.load(f'{path_prefix}_cleaner.joblib')
        instance.label_encoder = joblib.load(f'{path_prefix}_label_encoder.joblib')
        instance.scaler = joblib.load(f'{path_prefix}_scaler.joblib')
        instance.model = load_model(f'{path_prefix}_model.keras')
        return instance

# RF class

class RFPipeline:
    def __init__(self):
        self.cat_features = ['jour_semaine']
        self.num_features = ['jour_mois', 'annee', 'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos']
        self.compteur_col = ['nom_du_compteur']

        self.ohe_compteur = OneHotEncoder(drop='first', sparse_output=False)
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(drop='first'), self.cat_features),
            ('num', StandardScaler(), self.num_features)
        ])

        self.cleaner = Pipeline([
            ('raw', RawCleanerTransformer(keep_compteur=True)),
            ('feat', TimeFeatureTransformer())
        ])

        self.model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42)

    def _transform(self, X, fit=False):
        """Nettoyage + feature engineering + preprocessing complet."""
        X_clean = self.cleaner.fit_transform(X) if fit else self.cleaner.transform(X)
        X_base = self.preprocessor.fit_transform(X_clean) if fit else self.preprocessor.transform(X_clean)
        X_compteur = self.ohe_compteur.fit_transform(X_clean[self.compteur_col]) if fit else self.ohe_compteur.transform(X_clean[self.compteur_col])
        return np.hstack([X_base, X_compteur])

    def fit(self, X, y):
        X_concat = self._transform(X, fit=True)
        self.model.fit(X_concat, y)

    def predict(self, X):
        X_concat = self._transform(X, fit=False)
        return self.model.predict(X_concat)

    def preprocess(self, X):
        """Transformation sans prédiction (utile pour debug ou export)."""
        return self._transform(X, fit=False)

    def predict_clean(self, X):
        """Encapsulation propre pour prédiction depuis brut."""
        return self.predict(X)

    def save(self, path_prefix='rf_prod'):
        joblib.dump(self.cleaner, f'{path_prefix}_cleaner.joblib')
        joblib.dump(self.preprocessor, f'{path_prefix}_preprocessor.joblib')
        joblib.dump(self.ohe_compteur, f'{path_prefix}_ohe_compteur.joblib')
        joblib.dump(self.model, f'{path_prefix}_model.joblib')

    @classmethod
    def load(cls, path_prefix='rf_prod'):
        instance = cls()
        instance.cleaner = joblib.load(f'{path_prefix}_cleaner.joblib')
        instance.preprocessor = joblib.load(f'{path_prefix}_preprocessor.joblib')
        instance.ohe_compteur = joblib.load(f'{path_prefix}_ohe_compteur.joblib')
        instance.model = joblib.load(f'{path_prefix}_model.joblib')
        return instance
