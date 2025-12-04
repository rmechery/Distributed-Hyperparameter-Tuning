import pytest, sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.evaluate as eval 

def mock_message(score): # Mock Kafka messages
    msg = MagicMock()
    msg.value = {"score": score}
    return msg

@patch("src.evaluate.KafkaConsumer")
@patch("matplotlib.pyplot.savefig")  # prevent saving real file
def test_collect_all_and_plot(mock_savefig, mock_kafka):
    probe_inst, consumer_inst = MagicMock(), MagicMock() # Mock KafkaConsumer instance for each topic and for each probe
    probe_inst.topics.return_value = {"results_exp1", "results_exp2"}
    consumer_inst.__iter__.return_value = [mock_message(0.8), mock_message(0.6)]
    mock_kafka.side_effect = [probe_inst, consumer_inst, consumer_inst] 
    results = eval.collect_all(limit=2)
    
    assert len(results.items()) == 2
    assert "exp1" in results and "exp2" in results
    assert results["exp1"][0]["score"] == 0.8 and results["exp1"][1]["score"] == 0.6
    assert results["exp2"][0]["score"] == 0.8 and results["exp2"][1]["score"] == 0.6

    eval.plot_loss_curves(results) # Call plot_loss_curves with the results
    mock_savefig.assert_called_once_with("loss_curves.png")

@patch("src.evaluate.KafkaConsumer")
def test_only_collect_all(mock_kafka):
    probe_inst, consumer_inst = MagicMock(), MagicMock() # Mock KafkaConsumer instance for each topic and for each probe
    probe_inst.topics.return_value = {"results_exp1"}
    consumer_inst.__iter__.return_value = [mock_message(0.51)]
    mock_kafka.side_effect = [probe_inst, consumer_inst] 
    results = eval.collect_all(limit=1)
    assert len(results.items()) == 1 
    assert "exp1" in results and results["exp1"][0]["score"] == 0.51
        