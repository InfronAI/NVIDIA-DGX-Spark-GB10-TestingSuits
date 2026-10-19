"""Publication checks for deployment identities without logging matched values."""
import base64
import ipaddress
import re
from urllib.parse import urlsplit

# Documentary references are explicit. Endpoint examples use reserved names.
PUBLIC_DOMAINS = {
    'arxiv.org', 'creativecommons.org', 'nvidia.com', 'cloudflare.com',
    'sglang.ai', 'sglang.io', 'vllm.ai', 'github.com', 'githubusercontent.com',
    'huggingface.co', 'matplotlib.org', 'purl.org', 'apache.org',
    'globalpetrolprices.com', 'gutenberg.org', 'w3.org', 'infron.ai',
}
CODE_IDENTIFIERS = {'labels.net', 'n.local', 'r.info', 'i.info', 'n.info'}
DOMAIN = re.compile(r'(?i)\b(?:[a-z0-9][a-z0-9-]*\.)+(?:com|net|org|io|ai|cn|dev|co|gov|edu|internal|local|invalid|pro|app|cloud|tech|info)\b')
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(x) for x in (
    '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '100.64.0.0/10'))
DOCUMENTATION_NETWORKS = tuple(ipaddress.ip_network(x) for x in (
    '192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24'))
CREDENTIALS = [
    re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
]


def allowed_host(host):
    host = host.lower().rstrip('.')
    if host in {'*', 'localhost', 'serving.local', 'example.invalid'} or host.endswith('.invalid'):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return any(host == domain or host.endswith('.' + domain) for domain in PUBLIC_DOMAINS)
    return address.is_loopback or address.is_unspecified or any(address in net for net in DOCUMENTATION_NETWORKS if address.version == net.version)


def check_text(text, known_private_values=()):
    """Return category/line pairs; never emit credentials or network addresses.

    Known deployment values may be passed by a private publisher. Do not put
    those values in fixtures or checked-in configuration.
    """
    findings = set()

    def add(category, offset):
        findings.add((category, text.count('\n', 0, offset) + 1))

    for private in known_private_values:
        if private and private in text: add('private-value', text.index(private))
    for pattern in CREDENTIALS:
        for match in pattern.finditer(text): add('credential', match.start())
    # Inspect SVG payloads explicitly; embedded fonts/images are not hostnames.
    for match in re.finditer(r'data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)', text):
        try: decoded = base64.b64decode(match[1], validate=True).decode()
        except (ValueError, UnicodeError):
            add('invalid-embedded-svg', match.start()); continue
        if check_text(decoded, known_private_values): add('embedded-svg', match.start())
    searchable = re.sub(r'''data:[^\s"']{0,100};base64,[A-Za-z0-9+/=]+''', '<embedded-asset>', text)
    for match in re.finditer(r'''https?://[^\s<>"'`\\)\]]+''', searchable):
        try: host = urlsplit(match[0]).hostname
        except ValueError:
            add('invalid-url', match.start()); continue
        if host and '{' not in host and not allowed_host(host): add('unapproved-network-host', match.start())
    for match in DOMAIN.finditer(searchable):
        if match[0].lower().endswith('.app') and '/Applications/' in searchable[max(0, match.start() - 80):match.start()]:
            continue
        if match[0].lower() not in CODE_IDENTIFIERS and not allowed_host(match[0]): add('unapproved-bare-host', match.start())
    for match in IPV4.finditer(searchable):
        try: address = ipaddress.ip_address(match[0])
        except ValueError: continue
        if any(address == net.network_address and searchable[match.end():].startswith('/' + str(net.prefixlen)) for net in PRIVATE_NETWORKS):
            continue
        if any(address in net for net in PRIVATE_NETWORKS): add('private-ipv4', match.start())
    for match in re.finditer(r'/home/([A-Za-z0-9_.-]+)|/Users/([A-Za-z0-9_.-]+)', searchable):
        if (match[1] or match[2]) not in {'benchmark', 'user', 'USERNAME'}: add('personal-home', match.start())
    for match in re.finditer(r'\b([A-Za-z0-9_.-]+)@(?:spark-node[01]|\d{1,3}\.)', searchable):
        if match[1] not in {'benchmark', 'user', 'USERNAME'}: add('deployment-username', match.start())
    return [{'category': category, 'line': line} for category, line in sorted(findings)]
