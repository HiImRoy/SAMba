'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
REFACTOR: Updated to use the new build_model factory function.
'''

from thop import profile
import torch
from main import get_args_parser
import argparse
from models import build_model # Updated import

parser = argparse.ArgumentParser('SCSEGAMBA FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    # Use the new model factory function
    model, _ = build_model(args)
    model.to(args.device)

    input = torch.randn(1, 3, 512, 512)
    samples = input.to(torch.device(args.device))

    flops, params = profile(model, (samples, ))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
