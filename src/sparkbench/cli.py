"""Offline commands by default; live workloads require --execute."""
import argparse
import asyncio
import json
from pathlib import Path
from .registry import catalog,read,validate
from .analysis import write_summary
from .plans import length_matrix


def main():
    parser=argparse.ArgumentParser(prog='sparkbench')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['list','validate']:
        cmd=sub.add_parser(name);cmd.add_argument('--repo',default='.')
    cmd=sub.add_parser('summarize');cmd.add_argument('experiment');cmd.add_argument('--output',required=True)
    cmd=sub.add_parser('matrix');cmd.add_argument('--context',type=int,default=262144)
    cmd=sub.add_parser('serve');cmd.add_argument('--config',required=True);cmd.add_argument('--workloads',required=True);cmd.add_argument('--output',default='outputs/serving');cmd.add_argument('--execute',action='store_true')
    cmd=sub.add_parser('hardware');cmd.add_argument('--mode',choices=['base','extensions','collectives','overlap','soak'],default='base');cmd.add_argument('--output',default='outputs/hardware');cmd.add_argument('--rank',type=int,choices=[0,1],default=0);cmd.add_argument('--master-addr',default='127.0.0.1');cmd.add_argument('--master-port',type=int,default=18571);cmd.add_argument('--duration-limit',type=float,default=36000);cmd.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.command=='list':result=catalog(args.repo)
    elif args.command=='validate':result=validate(args.repo)
    elif args.command=='summarize':result=write_summary(args.experiment,args.output)
    elif args.command=='matrix':result=length_matrix(args.context)
    elif args.command=='serve':
        from .serving import run
        result=asyncio.run(run(read(args.config),args.workloads,args.output,args.execute))
    else:
        from .hardware import run
        result=run(args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
