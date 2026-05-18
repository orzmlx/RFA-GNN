import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from train_common import split_mean_logvar


class GenericPCCCallback(keras.callbacks.Callback):
    def __init__(
        self,
        train_data,
        val_data,
        evaluate_pack_fn,
        batch_size=32,
        max_eval=2048,
        extra_log_keys=None,
        scale_stats_fn=None,
        uncertainty_stats_fn=None,
        best_weights_path=None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.evaluate_pack_fn = evaluate_pack_fn
        self.batch_size = int(batch_size)
        self.max_eval = int(max_eval) if max_eval is not None else None
        self.extra_log_keys = list(extra_log_keys or [])
        self.scale_stats_fn = scale_stats_fn
        self.uncertainty_stats_fn = uncertainty_stats_fn
        self.best_weights_path = str(best_weights_path).strip() if best_weights_path is not None else ""
        self.best_epoch = None
        self.best_val_pcc = None
        self.best_val_mse = None

    def _evaluate(self, data_pack):
        return self.evaluate_pack_fn(
            self.model,
            data_pack,
            batch_size=self.batch_size,
            max_eval=self.max_eval,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tr = self._evaluate(self.train_data)
        va = self._evaluate(self.val_data)
        logs["mse"] = tr["mse"]
        logs["pcc"] = tr["pcc"]
        logs["val_mse"] = va["mse"]
        logs["val_pcc"] = va["pcc"]

        improved = self.best_val_pcc is None or float(va["pcc"]) > float(self.best_val_pcc)
        if improved:
            self.best_epoch = int(epoch) + 1
            self.best_val_pcc = float(va["pcc"])
            self.best_val_mse = float(va["mse"])
            if self.best_weights_path != "":
                out_dir = os.path.dirname(self.best_weights_path)
                if out_dir != "":
                    os.makedirs(out_dir, exist_ok=True)
                self.model.save_weights(self.best_weights_path)

        extra = []
        for key in self.extra_log_keys:
            value = logs.get(key)
            if value is None:
                continue
            try:
                extra.append(f"{key}={float(value):.4f}")
            except Exception:
                pass

        if self.scale_stats_fn is not None:
            try:
                scale_stats = self.scale_stats_fn(self.model) or {}
            except Exception:
                scale_stats = {}
            for key in ["target_scale", "cell_scale", "context_scale"]:
                value = scale_stats.get(key)
                if value is not None:
                    extra.append(f"{key}={float(value):.4f}")

        if self.uncertainty_stats_fn is not None:
            try:
                unc_stats = self.uncertainty_stats_fn(
                    self.model,
                    self.val_data,
                    batch_size=self.batch_size,
                    max_eval=self.max_eval,
                )
            except Exception:
                unc_stats = None
            if unc_stats is not None:
                for key in ["mu_mean", "mu_std", "logvar_mean", "logvar_min", "logvar_max", "sigma_mean"]:
                    value = unc_stats.get(key)
                    if value is not None:
                        extra.append(f"{key}={float(value):.4f}")

        prefix = (
            f"Epoch {epoch + 1}: "
            f"mse={tr['mse']:.4f} pcc={tr['pcc']:.4f} "
            f"val_mse={va['mse']:.4f} val_pcc={va['pcc']:.4f}"
        )
        print(prefix if not extra else prefix + " " + " ".join(extra))


def collect_gat_scale_stats(model):
    inner = getattr(getattr(model, "core", None), "gat", None)
    if inner is None:
        inner = getattr(model, "gat", None)
    if inner is None:
        return {}

    if hasattr(inner, "get_logged_scales"):
        scale_dict = inner.get_logged_scales()
        return {
            "target_scale": scale_dict.get("target_scale"),
            "cell_scale": scale_dict.get("cell_scale"),
            "context_scale": scale_dict.get("context_scale"),
        }

    stats = {}
    if hasattr(inner, "target_scale_logit"):
        stats["target_scale"] = float(tf.nn.softplus(inner.target_scale_logit).numpy())
    if hasattr(inner, "cell_scale_logit") and getattr(inner, "cell_scale_logit") is not None:
        stats["cell_scale"] = float(tf.nn.softplus(inner.cell_scale_logit).numpy())
    return stats


def build_uncertainty_stats_fn(loss_mask):
    valid_indices = np.where(np.asarray(loss_mask)[0] > 0)[0]

    def _uncertainty_stats_fn(model, data_pack, batch_size=32, max_eval=None):
        if not bool(getattr(model, "predict_uncertainty", False)):
            return None
        if len(data_pack) == 4:
            ctl, drug, cells, _y = data_pack
            drug_fp = None
        else:
            ctl, drug, cells, drug_fp, _y = data_pack
        if max_eval is not None and len(ctl) > int(max_eval):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(ctl), size=int(max_eval), replace=False)
            ctl = ctl[idx]
            drug = drug[idx]
            cells = cells[idx]
            if drug_fp is not None:
                drug_fp = drug_fp[idx]
        if drug_fp is None:
            pred = model.predict([ctl, drug, cells], batch_size=batch_size, verbose=0)
        else:
            pred = model.predict([ctl, drug, cells, drug_fp], batch_size=batch_size, verbose=0)
        mean_pred, logvar_pred = split_mean_logvar(pred)
        if logvar_pred is None:
            return None
        mean_pred = np.asarray(mean_pred, dtype=np.float32)
        clip_min = float(getattr(model, "logvar_clip_min", -6.0))
        clip_max = float(getattr(model, "logvar_clip_max", 2.0))
        logvar_pred = np.clip(np.asarray(logvar_pred, dtype=np.float32), clip_min, clip_max)
        mean_valid = mean_pred[:, valid_indices]
        logvar_valid = logvar_pred[:, valid_indices]
        sigma_valid = np.exp(0.5 * logvar_valid)
        return {
            "mu_mean": float(np.mean(mean_valid)),
            "mu_std": float(np.std(mean_valid)),
            "logvar_mean": float(np.mean(logvar_valid)),
            "logvar_min": float(np.min(logvar_valid)),
            "logvar_max": float(np.max(logvar_valid)),
            "sigma_mean": float(np.mean(sigma_valid)),
        }

    return _uncertainty_stats_fn


class GraphModelWrapper(keras.Model):
    def __init__(
        self,
        gat_model,
        graph_inputs,
        graph_input_dtypes,
        use_drug_fp_embedding=False,
        fp_table=None,
        pass_training=False,
    ):
        super().__init__()
        self.gat = gat_model
        self.graph_inputs = [
            tf.constant(value, dtype=dtype)
            for value, dtype in zip(graph_inputs, graph_input_dtypes)
        ]
        self.use_drug_fp_embedding = bool(use_drug_fp_embedding)
        self.fp_table = None if fp_table is None else tf.constant(fp_table, dtype=tf.float32)
        self.pass_training = bool(pass_training)

    def _resolve_drug_fp(self, drug_fp):
        if self.fp_table is not None and drug_fp.dtype.is_integer and len(drug_fp.shape) == 1:
            return tf.gather(self.fp_table, tf.cast(drug_fp, tf.int32))
        return drug_fp

    def call(self, inputs, training=False, **kwargs):
        if self.use_drug_fp_embedding:
            ctl, drug_targets, cell_idx, drug_fp = inputs
            drug_fp = self._resolve_drug_fp(drug_fp)
            model_inputs = list(self.graph_inputs) + [ctl, drug_targets, tf.cast(cell_idx, tf.int32), drug_fp]
        else:
            ctl, drug_targets, cell_idx = inputs
            model_inputs = list(self.graph_inputs) + [ctl, drug_targets, tf.cast(cell_idx, tf.int32)]
        if self.pass_training:
            kwargs["training"] = training
        return self.gat(model_inputs, **kwargs)
