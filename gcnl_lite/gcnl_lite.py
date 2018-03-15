# -*- coding: utf-8 -*-
# Copyright 2018 Pascual de Juan All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
__author__ = 'Pascual de Juan <pascual.dejuan@gmail.com>'
__version__ = '1.0'

"""
Mimics part of the Google Cloud Natural Language RESTful API using CoNLL 2017 models 
"""

import os
import tensorflow as tf
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder
from dragnn.python import spec_builder
from google.protobuf import text_format
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from tensorflow.python.platform import tf_logging as logging

import argparse
from ast import literal_eval
from flask import Flask, request, jsonify, abort

# This is because of the annoying warnings of the standard CPU TF distribution
app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # No TF optimization warnings
REPLS = ('attribute { name: ', ''), (' value: ', ': '), (' } ', ',\n')
lang = ""
segmenter_model = None
parser_model = None


def load_model(base_dir, master_spec_name, checkpoint_name):
    """Load the model from the base directory."""

    master_spec = spec_pb2.MasterSpec()  # Read the master spec
    with open(os.path.join(base_dir, master_spec_name), "r") as f:
        text_format.Merge(f.read(), master_spec)
    spec_builder.complete_master_spec(master_spec, None, base_dir)
    logging.set_verbosity(logging.WARN)  # Turn off TensorFlow spam.

    # Initialize a graph
    graph = tf.Graph()
    with graph.as_default():
        hyperparam_config = spec_pb2.GridPoint()
        builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
        # This is the component that will annotate test sentences.
        annotator = builder.add_annotation(enable_tracing=True)
        builder.add_saver()  # "Savers" can save and load models; here, we're only going to load.

    sess = tf.Session(graph=graph)
    with graph.as_default():
        builder.saver.restore(sess, os.path.join(base_dir, checkpoint_name))

    def annotate_sentence(sentence):
        with graph.as_default():
            return sess.run([annotator['annotations'], annotator['traces']],
                            feed_dict={annotator['input_batch']: [sentence]})

    return annotate_sentence


def annotate_text(text):
    """Do the actual parsing."""
    sentence = sentence_pb2.Sentence(
        text=text,
        token=[sentence_pb2.Token(word=text, start=-1, end=-1)]
    )

    # preprocess
    with tf.Session(graph=tf.Graph()) as tmp_session:
        char_input = gen_parser_ops.char_token_generator([sentence.SerializeToString()])
        preprocessed = tmp_session.run(char_input)[0]
    segmented, _ = segmenter_model(preprocessed)

    annotations, traces = parser_model(segmented[0])
    assert len(annotations) == 1
    assert len(traces) == 1
    return sentence_pb2.Sentence.FromString(annotations[0]), traces[0]


@app.route('/v1/documents:analyzeSyntax', methods=['POST'])
def documents_analyze_syntax():
    """Return a Google-style text parsing.

    The service follows the same structural interaction than the Google one
    but it uses the very tags of the CoNLL 2.0 specification. It lacks several
    out of scope features, like:

    * limited to one single language, therefore no auto-detection
    * ignored sentence break (periods, full stops...), treating all content as one single phrase
    * ignored "type" fixed to "PLAIN_TEXT",
    * ignored "encodingType" fixed to "UTF8",
    * skipped "sentiment" analysis returning "magnitude" and "score" always 0
    * adapted "partOfSpeech" keys and values to the FEATS CoNNL 2.0 convention
    * adapted "dependencyEdge" "label" to the DEPREL CoNNL 2.0 convention
    * skipped "lemma" value always to ""

    Request:
        {
            "document": {
                "type": "PLAIN_TEXT",
                "language": "es",
                "content": <the text to be analyzed>,
                },
            "encodingType": "UTF8"
        }

    Response:
        {
            "sentences": [
                {
                    "text": {
                        "content": <the analyzed text>,
                        "beginOffset": 0,
                    },
                    "sentiment": {
                        "magnitude": 0,
                        "score": 0,
                    },
                }
            ],
            "tokens": [
                {
                    "text": {
                        "content": <every word or part of a word>,
                        "beginOffset": <character offset from beginning>,
                    },
                    "partOfSpeech": {
                        "<key>": <value>,
                        "<key>": <value>,
                    },
                    "dependencyEdge": {
                        "headTokenIndex": <number reference to head token>,
                        "label": <kind of reference>,
                    },
                    "lemma": string
                }
            ],
            "language": "es"
        }
    """
    if not request.json and 'document' not in request.json and 'content' not in request.json['document']:
        abort(400)
    if request.json['document']['language'] != lang:
        error = jsonify({
            "error": {
                "code": 400,
                "message": "The language {} is not supported for syntax analysis.".format(
                    request.json['document']['language']),
                "status": "INVALID_ARGUMENT"
            }
        })
        error.status_code = 400
        return error
    parse_tree, _ = annotate_text(request.json['document']['content'])
    response = {
        "sentences": [
            {
                "text": {
                    "content": request.json['document']['content'],
                    "beginOffset": 0,
                },
                "sentiment": {
                    "magnitude": 0,
                    "score": 0,
                },
            }
        ],
        "tokens": [
        ],
        "language": lang
    }
    for idx, token in enumerate(parse_tree.token):
        response_token = {
            "text": {
                "content": token.word,
                "beginOffset": token.start,
            },
            "partOfSpeech": literal_eval('{{{}}}'.format(reduce(lambda a, kv: a.replace(*kv), REPLS, token.tag))),
            "dependencyEdge": {
                "headTokenIndex": token.head if token.head >= 0 else idx,
                "label": token.label,
            },
            "lemma": ""
        }
        response['tokens'].append(response_token)
    return jsonify(response)


def model_setup(base_lang, base_directory):  # Out of testing
    global lang
    global segmenter_model
    global parser_model

    lang = base_lang
    segmenter_model = load_model(
        os.path.join(base_directory, 'segmenter'),
        'spec.textproto', 'checkpoint')
    parser_model = load_model(
        base_directory,
        'parser_spec.textproto', 'checkpoint')


if __name__ == '__main__':
    """Start the HTTP RESTful service to mimic the Google Cloud Natural Language syntax analysis
    
    Args:
        lang: The language label to be accepted in the requests.
        dir: The base directory where the model resides in a CoNNL 17 structure. 

    Note:
        Checkpoint files need to be updated to Protocol Buffer using TF checkpoint_convert.py
    """
    parser = argparse.ArgumentParser(
        description='Mimics part of the Google Cloud Natural Language RESTful API using CoNLL 2017 models.')
    parser.add_argument("lang", type=str, choices=['en', 'es', 'fr', 'it', 'pt', 'de'],
                        help='language id')
    parser.add_argument('dir', type=str,
                        help="language resources base directory")
    parser.add_argument('-i', '--ip', type=str, default='0.0.0.0',
                        help='listen to the IP address')
    parser.add_argument('-p', '--port', type=int, default=7000,
                        help='listen to the port number')
    parser.add_argument('-X', '--debug', action="store_true",
                        help='debug mode')
    args = parser.parse_args()

    model_setup(args.lang, args.dir)
    app.run(host=args.ip, port=args.port, debug=args.debug)
