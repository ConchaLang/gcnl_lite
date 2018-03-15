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
Test _only_ the HTTP layer of the Google Cloud Natural Language 'light' RESTful API 
"""

import unittest
import gcnl_lite


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


class GcnlLiteTestCase(unittest.TestCase):
    def setUp(self):
        gcnl_lite.app.config['TESTING'] = True
        self.app = gcnl_lite.app.test_client()

    def test_analyze_syntax(self):
        rv = self.app.post(
            '/v1/documents:analyzeSyntax',
            data=REQUEST_1,
            mimetype='application/json'
        )
        self.assertTrue(rv.status_code == 200)
        self.assertTrue(b'"label": "root"' in rv.data)

    def test_analyze_syntax_2ndp_bad_request(self):
        rv = self.app.post(
            '/v1/documents:analyzeSyntax',
            data=REQUEST_2,
            mimetype='application/json'
        )
        self.assertTrue(rv.status_code == 400)
        self.assertTrue(b'The language ja is not supported for syntax analysis.' in rv.data)


gcnl_lite.model_setup(base_lang='es', base_directory='../../lang_models/es/')  # Out of testing.
if __name__ == '__main__':
    unittest.main()
