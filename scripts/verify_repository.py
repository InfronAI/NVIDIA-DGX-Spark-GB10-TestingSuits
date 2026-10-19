"""Offline publication checks: links, paired languages, resources and evidence."""
import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from sparkbench.registry import validate
from sparkbench.analysis import summarize
from sparkbench.publication import check_text

ROOT=Path(__file__).resolve().parents[1]

class ResourceParser(HTMLParser):
    def __init__(self):super().__init__();self.external=[];self.in_style=False;self.styles=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag in ('script','img','source') and attrs.get('src') and not attrs['src'].startswith('data:'):self.external.append(attrs['src'])
        if tag=='link' and attrs.get('href') and not attrs['href'].startswith('data:'):self.external.append(attrs['href'])
        if tag=='style':self.in_style=True
    def handle_endtag(self,tag):
        if tag=='style':self.in_style=False
    def handle_data(self,data):
        if self.in_style:self.styles.append(data)


def main():
    findings=validate(ROOT)
    imported=json.loads((ROOT/'provenance/import-manifest.json').read_text())
    for entry in imported['files']:
        target=(ROOT/entry['published']).resolve()
        assert target.is_relative_to(ROOT) and target.is_file(),entry['published']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['published_sha256'],entry['published']
    missing=[];secrets=[]
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ['.git','__pycache__','outputs']):continue
        if path.suffix=='.py':ast.parse(path.read_text())
        if path.suffix in ['.md','.json','.jsonl','.py','.txt','.html','.csv','.cu']:
            text=path.read_text()
            for issue in check_text(text):
                secrets.append(str(path.relative_to(ROOT))+':'+issue['category']+':'+str(issue['line']))
        if path.suffix=='.md':
            if '.zh.' in path.name and not path.with_name(path.name.replace('.zh.','.en.')).exists():missing.append(str(path)+':missing English pair')
            for href in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',text):
                if href.startswith(('http:','https:','#','mailto:','data:')):continue
                target=unquote(href.strip('<>').split('#')[0])
                if not (path.parent/target).exists():missing.append(str(path.relative_to(ROOT))+':'+target)
        if path.name.startswith('report.') and path.suffix=='.html':
            resources=ResourceParser();resources.feed(text)
            assert not resources.external,(path,resources.external)
            assert not re.search(r'url\([\"\']?(?!data:|#)[A-Za-z/]', ''.join(resources.styles)),path
    if missing or secrets:raise SystemExit(json.dumps({'missing_links':missing,'potential_secrets':sorted(set(secrets))},ensure_ascii=False,indent=2))
    result=summarize(ROOT/'experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark')
    print(json.dumps({'passed':True,**findings,'import_hashes_verified':len(imported['files']),'capacity_last':result['capacity'][-1],'evidence_complete':result['evidence_complete']},ensure_ascii=False))


if __name__=='__main__':main()
