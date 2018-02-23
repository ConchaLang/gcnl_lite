# Google Cloud Natural Language 'lite'
A reduced version of the official service which mimics part of 
the Google Cloud Natural Language RESTful API using CoNLL 2017 
models.

## Limitations
The service follows the same structural interaction than the 
[Google one](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeSyntax) 
but it uses the very tags of the CoNLL 2.0 specification. 
It lacks several out of scope features, like:

* limited to one single language, therefore no auto-detection
* ignored sentence break (periods, full stops...), treating all content as one single phrase
* ignored "type" fixed to "PLAIN_TEXT"
* ignored "encodingType" fixed to "UTF8"
* skipped "sentiment" analysis returning "magnitude" and "score" always 0
* adapted "partOfSpeech" keys and values to the FEATS CoNNL 2.0 convention
* adapted "dependencyEdge" "label" to the DEPREL CoNNL 2.0 convention
* skipped "lemma" value always to ""

Request:
```javascript
{
    "document": {
        "type": "PLAIN_TEXT",
        "language": "es",
        "content": <the text to be analyzed>,
        },
    "encodingType": "UTF8"
}
```
Response:
```javascript
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
```
## Installation
Due it relies in Syntaxnet Dragnn, which only runs on Python 2.7 
in a precise version of Tensorflow, there's a need to complile 
it locally according to the following instructions meanwhile a 
container version yet to come is in the works: 

### Installing virtualenv
```
> sudo -H pip install virtualenv
```

### Setting up virtual environments
```
> cd <virtualenvs_home>
> virtualenv -p python2.7 gcnl_lite2.7
```

### Activating and deactivating a virtual environment
```
> cd <virtualenvs_home>/gcnl_lite2.7
> source bin/activate
```

Once the virtual environment has been activated you can go 
whatever directory you want. All the pip installations will 
be hold in the activated virtual environment. In order to 
deactivate it just execute deactivate in whichever directory 
you are.

### Installing Bazel 0.5.4 for Tensorflow compilations
Tensorflow only compiles with Bazel 0.5.4 so you must install 
it in your virtual environment. Given Syntaxnet only works in 
Python 2.7, please activate gcnl_lite2.7 and follow the 
[download and installing instructions](https://docs.bazel.build/versions/master/install.html) 
as referenced in the Syntaxnet installation guide to 
download your OS version (remember MacOS is known as darwin). 
Then run this way to install it under the virtual environment:
```
> chmod +x <download>/bazel-0.5.4-installer-darwin-x86_64.sh
> <download>/bazel-0.5.4-installer-darwin-x86_64.sh --prefix=$VIRTUAL_ENV
```

### Installing Syntaxnet
Follow the [README.md](https://github.com/tensorflow/models/tree/master/research/syntaxnet#manual-installation) 
instructions to complete the manual installations and then do 
the build and test with the following commands:
```
> cd <projects_path>
> git clone --recursive https://github.com/tensorflow/models.git
> cd models/research/syntaxnet/tensorflow
> ./configure
> cd ..
> # On Mac, run the following:
> bazel test --linkopt=-headerpad_max_install_names dragnn/... syntaxnet/... util/utf8/...
```
Now it’s time to install with some variant to include Tensorflow.
```
> mkdir /tmp/syntaxnet_pkg
> bazel-bin/dragnn/tools/build_pip_package --include-tensorflow --output-dir=/tmp/syntaxnet_pkg
> sudo -H pip --no-cache-dir install /tmp/syntaxnet_pkg/syntaxnet_with_tensorflow-0.2-cp27-cp27m-macosx_10_6_intel.whl
```

There can be some extra import needs
```
> pip install backports.weakref
```

### Update checkpoint files
If you are willing to use the English and Spanish languages 
that are distributed in syntaxnet/examples/dragnn/data (for 
instance with the nearby *.ipynb notebooks) you MUST convert 
the checkpoint files to the appropriate protocol buffer 
version using `models/research/syntaxnet/tensorflow/tensorflow/contrib/rnn/python/tools/checkpoint_convert.py`

### Extra Languages
You can download up to 60 different language models 
[here](https://github.com/tensorflow/models/tree/master/research/syntaxnet/g3doc/conll2017).
All of them must get the mentioned checkpoint file updates.

## Running the service
You just have to run `python gcnl_lite.py en <wherever you 
left your English model>` or changing `en` per `es` if you 
have downloaded Spanish model. Service keeps on listening 
at port 7000. Use `curl` or PostMan to invoke the only 
endpoint `http://localhost:7000/v1/documents:analyzeSyntax`