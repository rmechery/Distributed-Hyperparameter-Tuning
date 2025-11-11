from unittest.mock import patch, MagicMock
import src.evaluate as eval

def test_mock_kafka():
    consumer = MagicMock()
    consumer.topics.return_value = {"results_1", "results_2"}
    msg1 = MagicMock()
    msg1.value = {"score": 0.8}
    msg2 = MagicMock()
    msg2.value = {"score": 0.11}
    consumer.__iter__.return_value = [msg1, msg2]

    with patch("src.evaluate.KafkaConsumer", return_value = consumer):
        data = eval.collect_all(limit = 2)

    assert "1" in data and isinstance(data["1"][0], dict) and data["1"][0]["score"] == 0.8 
    assert "2" in data and isinstance(data["2"][1], dict) and data["2"][1]["score"] == 0.11