from src.datathon_bank_marketing.data.download import download_data


def test_download_is_working():
    df = download_data()

    assert not df.empty
    assert "age" in df.columns
    assert "y" in df.columns