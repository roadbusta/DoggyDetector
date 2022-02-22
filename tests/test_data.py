from DoggyDetector.data import category_list

def test_len_category_list():
    assert len(category_list(DATADIR="raw_data/Images")) != 0
