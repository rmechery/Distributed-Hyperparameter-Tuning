from unittest.mock import patch, MagicMock
import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.consumer as consumer  

def test_consume_and_train():
    fake_params = {"model": "knn", "n_neighbors": 3}
    fake_msg = MagicMock()
    fake_msg.topic = "hyperparams_knn"
    fake_msg.value = fake_params

    mock_consumer = MagicMock() #Mocks the consumer to return the message
    mock_consumer.__iter__.return_value = [fake_msg]
    mock_probeconsumer = MagicMock() #Mock the probe consumer to return hyperparams_* topics
    mock_probeconsumer.topics.return_value = {"hyperparams_knn"}
    mock_prod = MagicMock() #Mock producer 

    #Patch KafkaConsumer twice: 1st: probe consumer (detects topics), 2nd call: worker consumer (returns messages)
    with patch("src.consumer.KafkaConsumer", side_effect=[mock_probeconsumer, mock_consumer]), \
         patch("src.consumer.KafkaProducer", return_value=mock_prod), \
         patch("src.consumer.train_model") as mock_train_model:

        mock_train_model.return_value = 0.85
        consumer.consume_and_train()

    mock_train_model.assert_called_with("knn", fake_params)   #mock training function
    mock_prod.send.assert_called_with("results_knn", value={"params": fake_params, "score": 0.85, "model": "knn"}) # output gets dispatched to Kafka results_knn

def test_no_topics_found(): #case of kafka having no hyperparams_* topics
    mock_probe = MagicMock()
    mock_probe.topics.return_value = set()
    with patch("src.consumer.KafkaConsumer", return_value=mock_probe):
        out = consumer.consume_and_train()
    assert out is None 
