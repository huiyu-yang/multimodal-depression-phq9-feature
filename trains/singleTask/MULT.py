import os
import time
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
import torch.nn.functional as F
import numpy as np
import sys


@torch.no_grad()
def compute_pf_mean_std(train_loader, device, pf_key="personal_feature", pf_dim=9):
    """Compute mean/std of personal_feature using ONLY train_loader."""
    s1 = torch.zeros(pf_dim, device=device)
    s2 = torch.zeros(pf_dim, device=device)
    n = 0

    for batch in train_loader:
        pf = batch[pf_key].to(device).float()  # [B, 9]
        s1 += pf.sum(dim=0)
        s2 += (pf * pf).sum(dim=0)
        n += pf.size(0)

    mean = s1 / max(n, 1)
    var = (s2 / max(n, 1)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    # avoid std=0 for constant dims in a fold
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std

def normalize_pf(pf, mean, std):
    """pf: [B,9] float tensor"""
    return (pf - mean) / std

logger = logging.getLogger('MSA')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(h)


class MULT():
    def __init__(self, args):
        self.args = args
        # self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        print(">>> ENTER do_train()")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-2
        )
        # optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=self.args.patience)

        self.args.pf_mean, self.args.pf_std = compute_pf_mean_std(
            train_loader=dataloader['train'],
            device=self.args.device,
            pf_key="personal_feature",
            pf_dim=9
        )
        print("PF mean:", self.args.pf_mean.detach().cpu().numpy())
        print("PF std :", self.args.pf_std.detach().cpu().numpy())

        def get_alpha(m):
            try:
                inner = m.Model
            except Exception:
                inner = m
            if hasattr(inner, "alpha"):
                return torch.sigmoid(inner.alpha).item()
            return None

        best_epoch = 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'  # now use F1_macro
        best_score = 1e8 if min_or_max == 'min' else -1e8  # CHANGED: best on VALID now

        for epochs in range(1, self.args.epochs + 1):
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    pf = batch_data['personal_feature'].to(self.args.device).float()  # (B, 9) modified by huiyu
                    pf = normalize_pf(pf, self.args.pf_mean, self.args.pf_std)

                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    # forward
                    outputs = model(text, audio, vision, pf)['M']
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    # backward
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            [param for param in model.parameters() if param.requires_grad],
                            self.args.grad_clip
                        )

                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            alpha_val = get_alpha(model)
            alpha_str = f" alpha={alpha_val:.4f}" if alpha_val is not None else ""
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f %s %s" % (
                self.args.modelName,
                epochs - best_epoch, epochs, self.args.cur_time, train_loss, dict_to_str(train_results), alpha_str
            ))

            # scheduler.step(train_loss)
            # ---------------- validation (every epoch) ----------------
            valid_results = self.do_test(model, dataloader['valid'], mode="VALID")
            cur_score = valid_results[self.args.KeyEval]
            isBetter = cur_score <= (best_score - 1e-6) if min_or_max == 'min' else cur_score >= (best_score + 1e-6)
            if isBetter:
                best_score, best_epoch = cur_score, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

        model.load_state_dict(torch.load(self.args.model_save_path))
        model.to(self.args.device)
        _ = self.do_test(model, dataloader['test'], mode="TEST")
        return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        conf_all = []
        conf_correct = []
        conf_wrong = []

        # ---- PF9 importance accumulators ----
        pf9_logit_sum = torch.zeros(9)  # sum over samples: |d logit_mdd / d pf|
        pf9_gate_sum  = torch.zeros(9)  # sum over samples: |d mean(gate) / d pf|
        pf9_count = 0

        has_pf_stats = hasattr(self.args, "pf_mean") and hasattr(self.args, "pf_std") \
                       and (self.args.pf_mean is not None) and (self.args.pf_std is not None)

        with tqdm(dataloader) as td:
            for i, batch_data in enumerate(td):
                vision = batch_data['vision'].to(self.args.device)
                audio  = batch_data['audio'].to(self.args.device)
                text   = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']['M'].to(self.args.device)

                if self.args.train_mode == 'classification':
                    labels = labels.view(-1).long()
                else:
                    labels = labels.view(-1, 1)

                if "personal_feature" in batch_data:
                    pf = batch_data['personal_feature'].to(self.args.device).float()  # [B, 9]
                    if has_pf_stats:
                        pf = normalize_pf(pf, self.args.pf_mean, self.args.pf_std)
                    else:
                        pf = pf / 2.0
                else:
                    pf = None

                with torch.no_grad():
                    if pf is None:
                        out = model(text, audio, vision)
                    else:
                        out = model(text, audio, vision, pf)
                    logits = out['M']

                loss = self.criterion(logits, labels)
                eval_loss += float(loss.item())

                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())

                # confidence
                if self.args.train_mode == 'classification':
                    probs = F.softmax(logits, dim=-1)
                    conf, pred_cls = probs.max(dim=-1)
                    conf_np = conf.detach().cpu().numpy()
                    conf_all.append(conf_np)

                    correct_mask = (pred_cls == labels).detach().cpu().numpy().astype(bool)
                    if correct_mask.any():
                        conf_correct.append(conf_np[correct_mask])
                    if (~correct_mask).any():
                        conf_wrong.append(conf_np[~correct_mask])

                # =========================
                # PF9 importance (temporarily enable grad)
                # =========================
                if (pf is not None) and (self.args.train_mode == "classification"):
                    # binary classification: we use class=1 (MDD) logit
                    with torch.enable_grad():
                        pf_req = pf.detach().requires_grad_(True)
                        out_imp = model(text, audio, vision, pf_req)
                        logits_imp = out_imp['M']  # (B, C)

                        if logits_imp.dim() == 2 and logits_imp.size(-1) >= 2:
                            # A) logit importance for MDD=1
                            target_logit = logits_imp[:, 1].mean()
                            grad_pf_logit = torch.autograd.grad(
                                target_logit, pf_req, retain_graph=True, create_graph=False
                            )[0]  # (B,9)

                            # B) gate-only importance (only if PF_gate_dim is returned)
                            if 'PF_gate_dim' in out_imp:
                                target_gate = out_imp['PF_gate_dim'].mean()
                                grad_pf_gate = torch.autograd.grad(
                                    target_gate, pf_req, retain_graph=False, create_graph=False
                                )[0]  # (B,9)
                            else:
                                grad_pf_gate = None

                            pf9_logit_sum += grad_pf_logit.detach().abs().sum(dim=0).cpu()
                            if grad_pf_gate is not None:
                                pf9_gate_sum += grad_pf_gate.detach().abs().sum(dim=0).cpu()
                            pf9_count += pf_req.size(0)

        # ---- finalize metrics ----
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        # ---- PF9 vectors ----
        if pf9_count > 0:
            pf9_imp_logit = (pf9_logit_sum / float(pf9_count)).numpy().tolist()
            pf9_imp_gate  = (pf9_gate_sum  / float(pf9_count)).numpy().tolist()
            eval_results["PF9_imp_logit_vec"] = pf9_imp_logit
            eval_results["PF9_imp_gate_vec"]  = pf9_imp_gate
            # command-line print
            print(f"[{mode}] PF9_imp_logit_vec = {[round(x, 6) for x in pf9_imp_logit]}")
            print(f"[{mode}] PF9_imp_gate_vec  = {[round(x, 6) for x in pf9_imp_gate]}")

        # ---- aggregate confidence stats ----
        if self.args.train_mode == 'classification' and len(conf_all) > 0:
            conf_all = np.concatenate(conf_all, axis=0)
            eval_results["Conf_mean"] = float(np.mean(conf_all))
            eval_results["Conf_p10"]  = float(np.quantile(conf_all, 0.10))
            eval_results["Conf_p50"]  = float(np.quantile(conf_all, 0.50))
            eval_results["Conf_p90"]  = float(np.quantile(conf_all, 0.90))

            if len(conf_correct) > 0:
                conf_correct = np.concatenate(conf_correct, axis=0)
                eval_results["Conf_correct_mean"] = float(np.mean(conf_correct))
            else:
                eval_results["Conf_correct_mean"] = float("nan")

            if len(conf_wrong) > 0:
                conf_wrong = np.concatenate(conf_wrong, axis=0)
                eval_results["Conf_wrong_mean"] = float(np.mean(conf_wrong))
            else:
                eval_results["Conf_wrong_mean"] = float("nan")

        log_results = {}
        for k, v in eval_results.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                log_results[k] = float(v)

        logger.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(log_results)))
        return eval_results
