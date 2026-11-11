import argparse
import json
from pathlib import Path

import numpy as np
import qlib
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import Feature, Ref
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData


DATE_SPLIT = {
    "train": ("2009-01-01", "2018-12-31"),
    "valid": ("2019-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2021-12-31"),
}


class SaveRollout(BaseCallback):
    def __init__(self, output, pool, test_calculator):
        super().__init__()
        self.output = output
        self.pool = pool
        self.test_calculator = test_calculator

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.logger.record("pool/size", self.pool.size)
        self.logger.record("pool/best_ic_ret", self.pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", self.pool.eval_cnt)
        if self.pool.size:
            ic, rank_ic = self.pool.test_ensemble(self.test_calculator)
            self.logger.record("test/ic", ic)
            self.logger.record("test/rank_ic", rank_ic)
        path = self.output / f"{self.num_timesteps}_steps"
        self.model.save(path)
        path.with_name(f"{path.name}_pool.json").write_text(
            json.dumps(self.pool.to_dict(), indent=2), encoding="utf-8"
        )


@torch.no_grad()
def evaluate(calculator, pool):
    score = calculator.make_ensemble_alpha(pool.exprs[:pool.size], pool.weights[:pool.size])
    target = calculator.target_value
    daily_ic = batch_pearsonr(score, target).cpu().numpy()
    # Rank IC uses pairwise comparisons, so evaluate it in date blocks.
    daily_rank_ic = np.concatenate([
        batch_spearmanr(values, returns).cpu().numpy()
        for values, returns in zip(score.split(64), target.split(64))
    ])
    ic, rank_ic = float(daily_ic.mean()), float(daily_rank_ic.mean())
    ic_std, rank_std = float(daily_ic.std()), float(daily_rank_ic.std())
    return {
        "ic": ic, "rank_ic": rank_ic, "ic_std": ic_std, "rank_ic_std": rank_std,
        "icir": ic / ic_std if ic_std else None,
        "rank_icir": rank_ic / rank_std if rank_std else None,
        "days": len(daily_ic),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AlphaGen on CSI300 with pool capacity 10.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qlib_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seed1"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250_000)
    args = parser.parse_args()

    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir()
    torch.set_num_threads(1)
    qlib.init(provider_uri=str(args.data_dir.expanduser().resolve()), region="cn", kernels=1)
    StockData._qlib_initialized = True
    reseed_everything(args.seed)

    device = torch.device("cuda:0")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    calculators = {
        split: QLibStockDataCalculator(StockData("csi300", start, end), target)
        for split, (start, end) in DATE_SPLIT.items()
    }
    pool = AlphaPool(capacity=10, calculator=calculators["train"], l1_alpha=0.005)
    model = MaskablePPO(
        "MlpPolicy", AlphaEnv(pool=pool, device=device, print_expr=True),
        policy_kwargs={
            "features_extractor_class": LSTMSharedNet,
            "features_extractor_kwargs": {"n_layers": 2, "d_model": 128, "dropout": 0.1, "device": device},
        },
        gamma=1.0, ent_coef=0.01, batch_size=128,
        tensorboard_log=str(out / "tensorboard"), device=device, verbose=1,
    )
    model.learn(total_timesteps=args.steps,
                callback=SaveRollout(checkpoints, pool, calculators["test"]),
                tb_log_name=f"csi300_pool10_seed{args.seed}")
    model.save(out / "final_model")
    (out / "final_pool.json").write_text(json.dumps(pool.to_dict(), indent=2) + "\n", encoding="utf-8")
    result = {
        "seed": args.seed, "requested_steps": args.steps, "actual_steps": model.num_timesteps,
        "pool_capacity": 10, "pool_size": pool.size, "date_split": DATE_SPLIT,
        "metrics": {split: evaluate(calculator, pool) for split, calculator in calculators.items()},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
