import os
import argparse
from utils.functions import Storage

class ConfigClassification():
    def __init__(self, args):
        HYPER_MODEL_MAP = {
            'mult': self.__MULT,
        }
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)

        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/path/to/datasets'
        tmp = {
            'cmdc': {
                'unaligned': {
                    'dataPath': '/path/to/CMDC_5fold_pkls/fold1_withPF.pkl',
                    'seq_lens': (1, 300, 300),
                    'feature_dims': (768, 40, 714),
                    'train_samples': 400,
                    'num_classes': 2,
                    'language': 'zh',
                    'KeyEval': 'F1_macro'
                }
            }
        }
        return tmp

    def __MULT(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'early_stop': 2,
                'use_bert': False,
                'use_bert_finetune': False,
                'attn_mask': True, 
                'update_epochs': 8,
            },
            'datasetParas':{
                'cmdc': {
                    'attn_dropout_a': 0.1,
                    'attn_dropout_v': 0.2,
                    'relu_dropout': 0.0,
                    'embed_dropout': 0.2,
                    'res_dropout': 0.0,
                    'dst_feature_dim_nheads': (30, 10),
                    'batch_size': 8,
                    'learning_rate': 1e-3,
                    'nlevels': 4,
                    'conv1d_kernel_size_l': 1,
                    'conv1d_kernel_size_a': 1,
                    'conv1d_kernel_size_v': 3,
                    'text_dropout': 0.4,
                    'attn_dropout': 0.2,
                    'output_dropout': 0.2,
                    'grad_clip': 1.0,
                    'patience': 20,
                    'weight_decay': 0.0,
                }
            },
        }
        return tmp

    def get_config(self):
        return self.args