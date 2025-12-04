from unittest.mock import patch, MagicMock
import sys, os, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.producer as producer
from src.search import random_search

def test_produce_configs():
    mock_producer = MagicMock() # Mock KafkaProducer instance
    with patch("src.producer.KafkaProducer", return_value=mock_producer):  # Patch KafkaProducer for mock instance
        producer.produce_configs(n_trials=2)
    
    assert mock_producer.send.call_count > 0  #producer.send was called at least once

    for topic in [call_args[0][0] for call_args in mock_producer.send.call_args_list]: # check if topics have correct names 
        assert topic.startswith("hyperparams_")
    
    for value in [call_args[1]['value'] for call_args in mock_producer.send.call_args_list]: 
        assert isinstance(value, dict) and "model" in value  # check if values are dictionaries with "model" key
    
    mock_producer.flush.assert_called_once()  # make sure flush was called

def test_produce_configs_skips_no_model(): # case without model key
    mock_producer = MagicMock()

    with patch("src.producer.KafkaProducer", return_value=mock_producer), \
         patch("src.producer.random_search", return_value=[{"n_neighbors": 5, "C": 1.0}]), \
         patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        producer.produce_configs(n_trials=1)

    assert "Skipping config without model key" in mock_stdout.getvalue()
    mock_producer.send.assert_not_called()  # producer.send isn't called
    mock_producer.flush.assert_called_once() # flush is called