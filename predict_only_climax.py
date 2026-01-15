# REMEMBER TO CHANGE THE NAMES OF THE VARS (LINE 33) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# example (activate climaX env)
# python predict_only_climax.py --ckpt /mnt/data/sonia/climax/1.40625deg.ckpt --res 1.40625 
#   --prompt_dir /mnt/data/sonia/climax-data/input-natlantic-windmag500hpa-fullcontext/val 
#   --batch 2 --out_path out/natlantic-windmag500hpa-fullcontext/val

import argparse, sys, numpy as np, torch, os
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Regional architecture (nn.Module, not Lightning)
from climax.regional_forecast.arch import RegionalClimaX

# ---- Variables list (order must match training!) ----
DEFAULT_VARS = [
    "land_sea_mask","orography","lattitude",
    "2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind",
    "geopotential_50","geopotential_250","geopotential_500",
    "geopotential_600","geopotential_700","geopotential_850","geopotential_925",
    "u_component_of_wind_50","u_component_of_wind_250","u_component_of_wind_500",
    "u_component_of_wind_600","u_component_of_wind_700","u_component_of_wind_850","u_component_of_wind_925",
    "v_component_of_wind_50","v_component_of_wind_250","v_component_of_wind_500",
    "v_component_of_wind_600","v_component_of_wind_700","v_component_of_wind_850","v_component_of_wind_925",
    "temperature_50","temperature_250","temperature_500","temperature_600",
    "temperature_700","temperature_850","temperature_925",
    "relative_humidity_50","relative_humidity_250","relative_humidity_500","relative_humidity_600",
    "relative_humidity_700","relative_humidity_850","relative_humidity_925",
    "specific_humidity_50","specific_humidity_250","specific_humidity_500","specific_humidity_600",
    "specific_humidity_700","specific_humidity_850","specific_humidity_925",
]
DEFAULT_USE_VARS = ["10m_u_component_of_wind","10m_v_component_of_wind"]

def parse_args():
    p = argparse.ArgumentParser(description="ClimaX predict-only (capture preds via metric)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", choices=["5.625","1.40625"], 
                   help="Picks img_size & patch_size to match ckpt")
    p.add_argument("--prompt_dir", help='path to directory with prompt .npy files')
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--use_vars", nargs="*", default=DEFAULT_USE_VARS)
    p.add_argument("--time_step", type=int, default=6, help="hourly, 6 hourly, etc")
    p.add_argument("--n_steps", type=int, default=7, help="number of time steps to predict")
    p.add_argument("--out_path", default='out')
    # core arch hyperparams (must match ckpt)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--drop_rate", type=float, default=0.1)
    return p.parse_args()

def geometry_for_resolution(res: str):
    if res == "5.625":      # 32x64, patch=2 -> 512 tokens
        return (32, 64), 2
    else:                   # 1.40625: 128x256, patch=4 -> 2048 tokens
        return (128, 256), 4

def build_region_info(H: int, W: int, patch: int):
    lat_vec = np.linspace(-90.0, 90.0, num=H, dtype=np.float32)
    lon_vec = np.linspace(0.0, 360.0, num=W, endpoint=False, dtype=np.float32)

    hp, wp = H // patch, W // patch
    L = hp * wp
    patch_ids = np.arange(L, dtype=np.int64)  # row-major [0..L-1]

    return {
        # full-domain index ranges (pixel/grid space)
        "min_h": 0, "max_h": H,          # NOTE: max_* is exclusive in most forks
        "min_w": 0, "max_w": W,

        # per-pixel indices (some code paths use these)
        "h_inds": np.arange(H, dtype=np.int64),
        "w_inds": np.arange(W, dtype=np.int64),

        # patch-level info (for token selection)
        "patch": patch,
        "hp": hp, "wp": wp,              # patches along H and W
        "patch_ids": patch_ids,          # all patches in row-major order

        # geo metadata (often optional but safe to include)
        "lat": lat_vec,
        "lon": lon_vec,
        "north": 90.0, "south": -90.0, "west": 0.0, "east": 360.0,
        "name": "full",
    }


def strip_prefixes(sd: dict) -> dict:
    for p in ("model.","net.","module.","climax."):
        if any(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k,v in sd.items()}
    return sd

def load_checkpoint(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    return strip_prefixes(sd)

class CaptureMetric:
    """Metric callable that captures y_pred and returns zero loss."""
    def __init__(self):
        self.last_pred = None
        self.last_logs = {}

    def __call__(self, preds: torch.Tensor, y_true: torch.Tensor,
                 out_vars, lat_vec: torch.Tensor):
        self.last_pred = preds
        # return (loss, logs_dict)
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype), {}


def main():
    args = parse_args()

    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    (H, W), patch = geometry_for_resolution(args.res)
    
    # load prompts 
    prompts = []
    sids = []
    for fname in os.listdir(args.prompt_dir):
        prompt = torch.from_numpy(np.load(os.path.join(args.prompt_dir, fname)))
        prompt = prompt.type(torch.float32)
        prompts.append(prompt)
        sids.append(fname.split('.')[0])
    dataset = TensorDataset(torch.stack(prompts))
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    # Build model
    model = RegionalClimaX(
        DEFAULT_VARS, (H, W), patch,
        args.embed_dim, args.depth, args.decoder_depth,
        args.num_heads, args.mlp_ratio, args.drop_path, args.drop_rate,
    )

    # Load weights (non-strict is fine if heads differ)
    sd = load_checkpoint(args.ckpt)
    sd = {k.replace("channel_embed", "var_embed"): v for k, v in sd.items()}
    sd = {k.replace("channel_query", "var_query"): v for k, v in sd.items()}
    sd = {k.replace("channel_agg", "var_agg"): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded state_dict | missing: {len(missing)} unexpected: {len(unexpected)}")

    # Dummy inputs (predict-only)
    B = args.batch
    x   = torch.randn(B, len(args.use_vars), H, W, dtype=torch.float32, device=device)
    y   = torch.zeros(B, len(args.use_vars), H, W, dtype=torch.float32, device=device)  # dummy target
    lat = torch.linspace(-90.0, 90.0, steps=H, dtype=torch.float32, device=device)
    # predict 6 hours ahead
    lead_times = torch.tensor([args.time_step], dtype=torch.float32, device=device)  # float avoids MPS Linear issues
    region_info = build_region_info(H, W, patch)
    
    os.makedirs(args.out_path, exist_ok=True)

    metric = CaptureMetric()

    model.to(device).eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds = batch[0].to(device)
            # if preds.ndim == 3:
            #     preds = preds.unsqueeze(0) # in case batch=1 and there's no batch dim
            step_preds = [preds]
            for _ in range(args.n_steps):
                # Many RegionalClimaX versions accept region_info at the end of forward(...).
                # Signature we satisfy: (x, y, lead_times, variables, out_variables, metric, lat, region_info)
                _ = model(preds, y, lead_times, args.use_vars, args.use_vars, [metric], lat, region_info)
                preds = metric.last_pred
                if preds is None:
                    print("Metric did not receive predictions; your forward signature may differ.", file=sys.stderr)
                    sys.exit(2)
                step_preds.append(preds)
            results.append(torch.stack(step_preds, dim=1).cpu()) # stacks time steps. (B, time, vars, H, W)
            # print(results[-1].shape)

    results = torch.cat(results, dim=0) # (N, time, vars, H, W)
    print(results.shape)
    for i in range(results.shape[0]):
        np.save(os.path.join(args.out_path, f"{sids[i]}"), results[i].numpy())

if __name__ == "__main__":
    main()
