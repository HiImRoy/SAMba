
# profile_model.py
import argparse
import torch
from thop import profile
import pandas as pd

# Local imports from your project
from models import build_model

def get_args_parser():
    """
    Defines and parses command-line arguments.
    This is a simplified version from main.py, focused on model loading.
    """
    parser = argparse.ArgumentParser('SAMbaCrack Profiler', add_help=False)

    # --- Model Parameters ---
    parser.add_argument('--model_name', default='SAMbaCrack', type=str, help="Name of the model to profile.")
    parser.add_argument('--training_stage', default=3, type=int, help="Training stage for the forward pass.")

    # --- Data Parameters ---
    parser.add_argument('--load_width', type=int, default=448, help="Image loading width.")
    parser.add_argument('--load_height', type=int, default=448, help="Image loading height.")

    # --- Environment Parameters (FIXED to use CUDA if available) ---
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for model & input during profiling.')

    # --- Dummy args to satisfy model constructor ---
    parser.add_argument('--hiera_depths', default=(2, 2, 6, 2), nargs='+', help="Depths for Hiera stages.")
    parser.add_argument('--hiera_num_heads', default=(2, 4, 8, 16), nargs='+', help="Num heads for Hiera stages.")
    parser.add_argument('--savss_drop_path_rate', default=0.1, type=float)
    parser.add_argument('--savss_use_rms_norm', default=True, type=bool)
    parser.add_argument('--savss_with_dwconv', default=True, type=bool)
    
    # --- Dummy args for loss criterion ---
    parser.add_argument('--BCELoss_ratio', default=0.83, type=float)
    parser.add_argument('--DiceLoss_ratio', default=0.17, type=float)

    return parser

def profile_model(model, dummy_input, training_stage, device):
    """
    Profiles the model and its main components to break down FLOPs.
    """
    results = {}

    # --- 1. Profile the entire model ---
    total_flops, total_params = profile(model, inputs=(dummy_input, training_stage), verbose=False)
    results['Total Model'] = {'flops': total_flops, 'params': total_params}

    # --- 2. Profile major components ---
    
    # Hiera (SAM) Encoder
    sam_flops, sam_params = profile(model.sam_encoder, inputs=(dummy_input,), verbose=False)
    results['Hiera Encoder'] = {'flops': sam_flops, 'params': sam_params}

    # --- SAVSS Encoder Components ---
    # a) Patch Embedding
    embed_flops, embed_params = profile(model.savss_patch_embed, inputs=(dummy_input,), verbose=False)
    
    # b) SAVSS Backbone Module (Mamba Layers)
    savss_input_token = torch.randn(1, (args.load_height // 8) ** 2, 256).to(device)
    hw_shape = (args.load_height // 8, args.load_width // 8)
    backbone_module_flops, backbone_module_params = profile(model.savss_backbone_module, inputs=(savss_input_token, hw_shape), verbose=False)

    # c) Neck Projections
    neck_flops, neck_params = 0, 0
    neck_input_dummy = torch.randn(1, 256, 56, 56).to(device)
    for neck_proj in model.savss_neck:
        proj_flops, proj_params = profile(neck_proj, inputs=(neck_input_dummy,), verbose=False)
        neck_flops += proj_flops
        neck_params += proj_params

    results['SAVSS Encoder'] = {
        'flops': embed_flops + backbone_module_flops + neck_flops,
        'params': embed_params + backbone_module_params + neck_params
    }

    # --- Fusion and Decoder Components ---
    fusion_flops, fusion_params = 0, 0
    with torch.no_grad():
        sam_features = model.sam_encoder(dummy_input)
    
    for i, (adapter, refiner) in enumerate(zip(model.adapters, model.refiners)):
        feature = sam_features[i]
        adapter_flops, adapter_params = profile(adapter, inputs=(feature,), verbose=False)
        refiner_flops, refiner_params = profile(refiner, inputs=(feature,), verbose=False)
        fusion_flops += adapter_flops + refiner_flops
        fusion_params += adapter_params + refiner_params

    # b) HOACM (Fusion)
    mamba_features_dummy = [torch.randn_like(f) for f in sam_features]
    for i, hoacm in enumerate(model.hoacms):
        hoacm_flops, hoacm_params = profile(hoacm, inputs=(sam_features[i], mamba_features_dummy[i]), verbose=False)
        fusion_flops += hoacm_flops
        fusion_params += hoacm_params

    # c) Decoder
    decoder_input_dummy = (sam_features[3], sam_features[2], sam_features[1], sam_features[0])
    decoder_flops, decoder_params = profile(model.decoder, inputs=(decoder_input_dummy,), verbose=False)
    
    results['Fusion & Decoder'] = {
        'flops': fusion_flops + decoder_flops,
        'params': fusion_params + decoder_params
    }

    return results

def print_report(results):
    """Prints a formatted report from the profiling results."""
    
    total_flops = results['Total Model']['flops']
    
    df_data = []
    for name, values in results.items():
        if name == 'Total Model':
            continue
        flops = values['flops']
        params = values['params']
        percentage = (flops / total_flops) * 100
        df_data.append({
            "Module": name,
            "FLOPs (G)": f"{flops / 1e9:.2f}",
            "Params (M)": f"{params / 1e6:.2f}",
            "Percentage (%)": f"{percentage:.2f}%"
        })

    df = pd.DataFrame(df_data)
    
    print("\n--- Model Computation Analysis ---")
    print(f"Input Size: ({args.load_height}x{args.load_width})\n")
    print(df.to_string(index=False))
    
    total_params = results['Total Model']['params']
    print("\n" + "-"*50)
    print(f"Total FLOPs: {total_flops / 1e9:.2f} G")
    print(f"Total Params: {total_params / 1e6:.2f} M")
    print("-" * 50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAMbaCrack Profiler', parents=[get_args_parser()])
    args = parser.parse_args()

    # Build the model
    model, _ = build_model(args)

    # [FIX] Explicitly move the model to the correct device (CUDA if available)
    model.to(args.device)
    model.eval()

    # Create a dummy input tensor and move it to the same device
    dummy_input = torch.randn(1, 3, args.load_height, args.load_width).to(args.device)

    # Profile the model and its components
    analysis_results = profile_model(model, dummy_input, args.training_stage, args.device)

    # Print the final report
    print_report(analysis_results)

