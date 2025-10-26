# tests/conftest.py
import types
import numpy as np
import pytest

class FakeEncoder:
    def encode(self, texts):
        # 384 to częsty rozmiar 'all-MiniLM-L6-v2', ale nie ma to znaczenia dla testów
        return np.zeros((len(texts), 384), dtype=float)

class FakeChromaCollection:
    def __init__(self, n=5):
        self.n = n
    def query(self, query_embeddings=None, n_results=5, **kwargs):
        docs = [f"similar doc {i}" for i in range(self.n)]
        metas = [{"price": float(10 * (i+1))} for i in range(self.n)]
        return {"documents":[docs], "metadatas":[metas]}

class FakeLRModel:
    def predict(self, X):
        # Zwraca średnią z trzech bazowych przewidywań, jeśli dostępne
        cols = X.columns
        s = X[cols].mean(axis=1).values
        return s

class FakeRFModel:
    def predict(self, X):
        return np.array([123.0])  # stała predykcja (sprawdza ścieżkę bez LLM)

@pytest.fixture
def fake_encoder():
    return FakeEncoder()

@pytest.fixture
def fake_collection():
    return FakeChromaCollection()

@pytest.fixture
def fake_lr_model():
    return FakeLRModel()

@pytest.fixture
def fake_rf_model():
    return FakeRFModel()
