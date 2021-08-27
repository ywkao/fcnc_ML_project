try:
    import tensorflow.keras as keras
except:
    import keras as keras
keras.backend.set_learning_phase(0) # you *NEED* this line, otherwise your DNN will crash in CMSSW/tensorflow C++ API
				    # this line tells tensorflow to save the graph in "prediction" mode rather than "learning" mode
import tensorflow as tf

def load_model(config, weights):
    with open(config, "r") as f_in:
      model = keras.models.model_from_json(f_in.read())
    model.load_weights(weights)
    return model 

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



# main

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_hdf5", help = "input weights file", type=str)
parser.add_argument("--input_json", help = "input config file", type=str)
args = parser.parse_args()

model = load_model(args.input_json, args.input_hdf5)

frozen_graph = freeze_session(keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])

with open(args.input_hdf5.replace(".hdf5", "_nodes.txt"), "w") as f_out:
  nodes =  [n.name for n in tf.get_default_graph().as_graph_def().node]
  for node in nodes:
    f_out.write(node + "\n")

tf.train.write_graph(frozen_graph, "", args.input_hdf5.replace(".hdf5", ".pb"), as_text=False)
