from app.predict import predict_species


def test_predict_species():
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict_species(sample)

    assert result in ["setosa", "versicolor", "virginica"]
