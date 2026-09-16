from src.datathon_bank_marketing.data.download import load_data


def test_download_is_working():
    df = load_data()

    assert not df.empty
    assert "age" in df.columns
    assert "y" in df.columns