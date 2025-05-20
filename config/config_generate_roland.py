import yaml
import os

def generate_full_config(params: dict) -> dict:
    """生成完整的配置字典"""
    return {
        "print": "both",
        "device": "auto",
        "out_dir": "results",
        "cfg_dest": "config.yaml",
        "seed": 1,
        "round": 4,
        "tensorboard_each_run": False,
        "tensorboard_agg": True,
        "num_workers": 0,
        "num_threads": 6,
        "metric_best": "auto",
        "view_emb": False,

        "dataset": {
            "name": params.get("dataset_name", "bitcoin-alpha"),
            "task": "link_pred",
            "task_type": "classification",
            "task_splitting": params.get("task_splitting", "within"),
            "split": params.get("split", [0.8, 0.1, 0.1]),
            "edge_encoder": True,
            "edge_encoder_bn": True,
            "edge_dim": 128
        },

        "train": {
            "mode": params.get("train_mode", "live_update_topo"),
            "epoch_resume": -1,
            "ckpt_clean": True,
            "stop_live_update_after": 9999999,
            "internal_validation_tolerance": params.get("internal_validation_tolerance", 5),
            "memory_steps": params.get("memory_steps", 3)
        },

        "model": {
            "type": params.get("model_type", "roland"),
            "loss_fun": "cross_entropy",
            "size_average": True,
            "thresh": 0.5,
            "edge_decoding": "concat"
        },

        "gnn": {
            "layers_mp": params.get("layers_mp", 2),
            "dim_inner": params.get("dim_inner", 256),
            "layer_type": params.get("layer_type", "residual_edge_conv"),
            "batchnorm": True,
            "act": "prelu",
            "dropout": 0.0,
            "agg": "add",
            "normalize_adj": False,
            "msg_direction": "single",
            "att_heads": 1,
            "l2norm": True,
            "skip_connection": "affine"
        },

        "optim": {
            "optimizer": "adam",
            "base_lr": params.get("base_lr", 0.005),
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "scheduler": "cos",
            "meta_lr": params.get("meta_lr", 0.1),
            "meta_weight_decay": 0.0005,
            "steps": [30, 60, 90],
            "lr_decay": 0.1,
            "max_epoch": params.get("max_epoch", 100)
        },

        "bn": {
            "eps": 1e-5,
            "mom": 0.1
        },

        "mem": {
            "inplace": False
        },

        "experimental": {
            "rank_eval_multiplier": 1000
        },

        "transaction": {
            "snapshot": False,
            "snapshot_num": 100,
            "snapshot_freq": "D",
            "check_snapshot": False,
            "history": "full",
            "loss": "meta",
            "feature_int_dim": 32,
            "feature_node_int_num": [0],
            "feature_amount_dim": 64,
            "feature_time_dim": 64,
            "node_feature": "raw",
            "horizon": 1,
            "pred_mode": "before",
            "time_enc_periods": [1],
            "time_enc_mode": "enc_before_diff",
            "keep_ratio": "linear"
        },

        "metric": {
            "mrr_method": "max"
        },

        "link_pred_spec": {
            "forecast_frequency": 1,
            "forecast_horizon": 7
        },

        "roland": {
            "is_meta": params.get("is_meta", True),
            "method": params.get("method", "moving_average"),
            "alpha": params.get("alpha", 0.8)
        },

        "windows": {
            "window_size": params.get("window_size", 7),
            "maml_lr": params.get("maml_lr", 0.006),
            "drop_rate": params.get("drop_rate", 0.4),
            "beta": params.get("beta", 0.5)
        },

        "topo": {
            "use_topo": params.get("use_topo", True),
            "meta_type": params.get("meta_type", "Residual"),
            "resolution": params.get("resolution", 50),
            "filtration": params.get("filtration", [[1,1]]),
            "remove_edge": params.get("remove_edge", "off"),
            "remove_ratio": params.get("remove_ratio", 1.0),
            "window_size": params.get("topo_window_size", 10000),
            "dropout": params.get("dropout", 0.2),
            "bandwidth": params.get("bandwidth", 2.0),
            "power": params.get("power", 2.0),
            "is_directed": params.get("is_directed", False),
            "distance": params.get("distance", "wasserstein"),
            "gamma": params.get("gamma", 0.1),
            "weight_method": params.get("weight_method", "exp"),
            "delta": params.get("delta", 0.0),
            "drop_rate": params.get("drop_rate", 0.2)
        }
    }

def generate_configs(output_dir: str = "config"):
    """生成所有配置组合的配置文件"""
    os.makedirs(output_dir, exist_ok=True)
    dataset_names = ['as', 'bitcoin-alpha', 'bitcoin-otc', 'uci-msg', 'reddit-body', 'reddit-title', 'ethereum']
    remarks = ['as_topo', 'alpha_topo', 'otc_topo', 'uci_topo', 'body_topo', 'title_topo', 'ethereum_topo']
    remark_list = ["alpha_topo", "otc_topo", "uci_topo", "body_topo", "title_topo", "ethereum_topo"]
    os.makedirs(os.path.join(output_dir, "roland_example"), exist_ok=True)
    for remark in remark_list:
        dataset_idx = remarks.index(remark)
        dataset_name = dataset_names[dataset_idx]
        params = {
            "dataset_name": dataset_name,
            }
        config = generate_full_config(params)
        filename = f"{remark}.yaml"
        filepath = os.path.join(output_dir, "roland_example", filename)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
if __name__ == "__main__":
    generate_configs('./config')