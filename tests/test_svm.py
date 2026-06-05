import pytest
import pickle
import numpy as np
import pandas as pd

@pytest.fixture
def load_artifacts():
    """Fixture untuk memuat model dan scaler SVM sebelum pengujian dijalankan"""
    with open('models/svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/svm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def test_artifacts_loading(load_artifacts):
    """Memastikan file model dan scaler tidak kosong dan berhasil dimuat"""
    model, scaler = load_artifacts
    assert model is not None
    assert scaler is not None

def test_svm_prediction_output_format(load_artifacts):
    """Memastikan output dari model mengembalikan format data yang konsisten (0 atau 1)"""
    model, scaler = load_artifacts
    
    # Buat data dummy satu baris dengan jumlah fitur tiruan yang sesuai dengan model
    # Ganti angka 10 di bawah dengan total jumlah fitur akhir hasil preprocessing kamu
    dummy_input = np.random.rand(1, 10) 
    
    # Tembak ke fungsi transformasi dan prediksi
    dummy_scaled = scaler.transform(dummy_input)
    prediction = model.predict(dummy_scaled)
    prediction_proba = model.predict_proba(dummy_scaled)
    
    # Validasi Hasil (Assertion)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1], "Output prediksi harus berupa kelas binari: 0 atau 1"
    assert prediction_proba.shape == (1, 2), "Output probabilitas harus memiliki format 2 kelas (stay/leave)"
    assert 0.0 <= prediction_proba[0][1] <= 1.0, "Nilai probabilitas harus berada di rentang rentang 0 sampai 1"