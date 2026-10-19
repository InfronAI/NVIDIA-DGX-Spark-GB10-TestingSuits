import base64
import unittest

from sparkbench.publication import check_text


class Publication(unittest.TestCase):
    def test_bare_deployment_host_and_url_are_both_detected(self):
        host = '.'.join(('gpu', 'private', 'net'))
        self.assertTrue(check_text(host))
        self.assertTrue(check_text('https://' + host + '/v1'))

    def test_reserved_examples_and_public_documentation_are_allowed(self):
        text = '\n'.join(['https://docs.nvidia.com', 'https://llm.example.invalid', 'http://127.0.0.1:8000', 'benchmark@192.0.2.11', '/home/benchmark'])
        self.assertEqual(check_text(text), [])

    def test_secondary_private_network_and_tailnet_are_detected(self):
        self.assertTrue(check_text('.'.join(('192', '168', '101', '21'))))
        self.assertTrue(check_text('.'.join(('100', '78', '1', '2'))))

    def test_sdk_versions_are_not_mistaken_for_private_addresses(self):
        self.assertEqual(check_text('cuDNN 9.20.0.48; cuBLAS 13.1.1.3-1'), [])

    def test_library_properties_and_application_bundles_are_not_hosts(self):
        self.assertEqual(check_text('r.info=i.info; /Applications/Google Chrome.app/Contents'), [])
        self.assertTrue(check_text('https://' + 'r.info'))

    def test_home_without_trailing_slash_and_ssh_user_are_detected(self):
        self.assertTrue(check_text('/home/' + 'operator'))
        self.assertTrue(check_text('operator' + '@' + 'spark-node0'))

    def test_embedded_svg_is_inspected(self):
        svg = '<svg><text>' + '.'.join(('gpu', 'private', 'net')) + '</text></svg>'
        html = '<img src="data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode() + '">'
        self.assertTrue(check_text(html))

    def test_secret_matching_does_not_return_the_secret(self):
        value = 'sk-' + 'x' * 32
        result = check_text(value)
        self.assertTrue(result)
        self.assertNotIn(value, str(result))
        self.assertTrue(check_text('private deployment value', ['deployment value']))


if __name__ == '__main__': unittest.main()
