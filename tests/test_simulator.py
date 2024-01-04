from src.simulator import generate_client_ids

def test_generate_client_ids():
    ids = generate_client_ids(5)
    assert isinstance(ids, list)
    assert ids[0] == '0000'
    assert ids[3] == '0003'

