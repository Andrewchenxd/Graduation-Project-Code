"""
评估方案
在 RMLA、RMLB、RMLC 三个数据集的 test 集上评估模型，
输出 Overall Accuracy (OA) 以及每个 SNR 下的准确率。
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from functools import partial
import warnings

from model.ACSF_TMAE import ACSF_TMAE
from tool.rml2016a import SigDataSet_pwvd_fc
from tool.utils import load_checkpoint

warnings.filterwarnings("ignore")


class EvaluateScheme:
    """
    评估方案
    在指定数据集的测试集上评估模型性能
    """

    def __init__(self, args=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args

    def build_model(self, num_classes=11):
        """构建 ACSF_TMAE 模型实例"""
        patch_size = 4
        window_size = 2
        model = ACSF_TMAE(
            img_size=128, patch_size=patch_size, in_chans=1,
            decoder_embed_dim=384,
            depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=window_size, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            mask_ratio=0.75, mask_type='suiji',
            len_attn=[192, 256], task='classification', numclass=num_classes
        )
        return model

    def evaluate(self, model, test_loader):
        """
        在 test_loader 上评估模型

        返回:
            oa: Overall Accuracy
            snr_values: 所有 SNR 值
            snr_acc: 每个 SNR 对应的准确率
        """
        model.eval()
        model.to(self.device)

        all_preds = []
        all_labels = []
        all_snrs = []

        with torch.no_grad():
            for images, labels, snrs in tqdm(test_loader, desc="评估中"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images, images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_snrs.append(snrs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_snrs = np.concatenate(all_snrs)

        # 总体准确率
        total_correct = (all_preds == all_labels).sum()
        total_samples = len(all_labels)
        oa = total_correct / total_samples if total_samples > 0 else 0.0

        # 按 SNR 统计准确率
        unique_snrs = np.sort(np.unique(all_snrs))
        snr_acc = np.zeros(len(unique_snrs))
        for i, snr_val in enumerate(unique_snrs):
            mask = (all_snrs == snr_val)
            snr_correct = (all_preds[mask] == all_labels[mask]).sum()
            snr_total = mask.sum()
            snr_acc[i] = snr_correct / snr_total if snr_total > 0 else 0.0

        return oa, unique_snrs, snr_acc

    def evaluate_dataset(self, dataset_name, checkpoint_path, num_classes=11, batch_size=512):
        """在指定数据集的 test 集上评估"""
        print(f"\n{'='*60}")
        print(f"评估数据集: {dataset_name}")
        print(f"检查点: {checkpoint_path}")
        print(f"{'='*60}")

        test_data_path = f'./data/fintune_data/{dataset_name}_test_data.npy'
        print(f"加载测试数据: {test_data_path}")

        test_set = SigDataSet_pwvd_fc(
            test_data_path,
            newdata=False,
            adsbis=False,
            resample_is=False,
            samplenum=6,
            resize_is=False,
            norm='maxmin',
            snr_range=[10, 20],
            aug=False
        )

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            prefetch_factor=20
        )

        model = self.build_model(num_classes=num_classes)
        model = load_checkpoint(model, checkpoint_path, self.device)

        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU")
            model = nn.DataParallel(model)

        oa, snr_values, snr_acc = self.evaluate(model, test_loader)

        print(f"\n--- {dataset_name} 评估结果 ---")
        print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
        print(f"\n各 SNR 准确率:")
        print(f"{'SNR':<8} {'Accuracy(%)':<12}")
        print("-" * 20)
        for snr_val, acc in zip(snr_values, snr_acc):
            print(f"{snr_val:<8} {acc * 100:<12.2f}")

        return oa, snr_values, snr_acc

    def run(self):
        """在 RMLA、RMLB、RMLC 三个数据集上依次评估"""
        num_classes = 11
        batch_size = 512

        datasets = [
            {'name': 'RMLA', 'checkpoint': './checkpoint_fintune/RMLA/pwvd_best_network.pth'},
            {'name': 'RMLB', 'checkpoint': './checkpoint_fintune/RMLB/pwvd_best_network.pth'},
            {'name': 'RMLC', 'checkpoint': './checkpoint_fintune/RMLC/pwvd_best_network.pth'},
        ]

        results = {}
        for ds in datasets:
            oa, snr_values, snr_acc = self.evaluate_dataset(
                dataset_name=ds['name'],
                checkpoint_path=ds['checkpoint'],
                num_classes=num_classes,
                batch_size=batch_size
            )
            results[ds['name']] = {
                'OA': oa,
                'SNR_Values': snr_values,
                'SNR_Accuracy': snr_acc
            }

        # 汇总结果
        print(f"\n\n{'='*60}")
        print("最终汇总结果")
        print(f"{'='*60}")
        print(f"{'Dataset':<10} {'OA(%)':<10}")
        print("-" * 20)
        for name, res in results.items():
            print(f"{name:<10} {res['OA']*100:<10.2f}")

        # 各 SNR 准确率汇总表
        print(f"\n各 SNR 准确率汇总:")
        all_snr_vals = set()
        for res in results.values():
            all_snr_vals.update(res['SNR_Values'].tolist())
        all_snr_vals = sorted(all_snr_vals)

        header = f"{'Dataset':<10}"
        for sv in all_snr_vals:
            header += f"{f'SNR{sv}':<10}"
        print(header)
        print("-" * len(header))

        for name, res in results.items():
            snr_map = dict(zip(res['SNR_Values'], res['SNR_Accuracy']))
            row = f"{name:<10}"
            for sv in all_snr_vals:
                acc = snr_map.get(sv, None)
                if acc is not None:
                    row += f"{acc * 100:<10.2f}"
                else:
                    row += f"{'N/A':<10}"
            print(row)

        return results
