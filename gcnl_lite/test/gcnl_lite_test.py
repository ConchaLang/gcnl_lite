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

import gcnl_lite
import os
import unittest


REQUEST_1 = """{
  "document": {
    "type": "PLAIN_TEXT",
        "language": "es",
    "content": "Concha dice la verdad"
  },
    "encodingType": "UTF8"
}"""

REQUEST_2 = """{
  "document": {
    "type": "PLAIN_TEXT",
        "language": "ja",
    "content": "Concha dice la verdad"
  },
    "encodingType": "UTF8"
}"""


class ServersTestCase(unittest.TestCase):
    def setUp(self):
        gcnl_lite.lang = 'es'
        base_directory = '../../lang_models/es/'
        gcnl_lite.segmenter_model = gcnl_lite.load_model(
            os.path.join(base_directory, 'segmenter'),
            'spec.textproto', 'checkpoint')
        gcnl_lite.parser_model = gcnl_lite.load_model(
            base_directory,
            'parser_spec.textproto', 'checkpoint')
        gcnl_lite.app.config['TESTING'] = True
        self.app = gcnl_lite.app.test_client()

    def test_01_analyze_syntax(self):
        rv = self.app.post(
            '/v1/documents:analyzeSyntax',
            data=REQUEST_1,
            mimetype='application/json'
        )
        self.assertTrue(rv.status_code == 200)
        self.assertTrue(b'"label": "root"' in rv.data)
        # Secondary path (bad request)
        rv = self.app.post(
            '/v1/documents:analyzeSyntax',
            data=REQUEST_2,
            mimetype='application/json'
        )
        self.assertTrue(rv.status_code == 400)
        self.assertTrue(b'The language ja is not supported for syntax analysis.' in rv.data)


if __name__ == '__main__':
    unittest.main()
