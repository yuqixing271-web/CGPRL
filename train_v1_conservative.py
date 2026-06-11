import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".."); sys.path.insert(0, "../..")

import logging, torch, numpy as np, random
from utils import create_logger, copy_all_src
from CrossProblemICLTrainer import CrossProblemICLTrainer as Trainer

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ================================================================
# 32 种 VRP 类型定义
# ================================================================
TRAIN_TYPES = [
    'CVRP',
    'VRPB', 'OVRP', 'VRPL', 'VRPTW', 'VRPK',
    'OVRPB', 'VRPBL', 'VRPBTW', 'VRPBK',
    'OVRPL', 'OVRPTW', 'OVRPK',
    'VRPLTW', 'VRPLK', 'VRPTWK',
]
TEST_TYPES = [
    'OVRPBL', 'OVRPBTW', 'OVRPBK', 'VRPBLTW', 'VRPBLK',
    'VRPBTWK', 'OVRPLTW', 'OVRPLK', 'OVRPTWK', 'VRPLTWK',
    'OVRPBLTW', 'OVRPBLK', 'OVRPBTWK', 'VRPBLTWK', 'OVRPLTWK',
    'OVRPBLTWK',
]

# 加权采样配置 
TRAIN_TYPE_WEIGHTS = {
    'CVRP': 2.0,
    'VRPB': 5.0,   'VRPTW': 5.0,
    'OVRP': 3.0,   'VRPL': 3.0,  'VRPK': 3.0,
    'OVRPB': 1.0,  'VRPBL': 1.0, 'VRPBTW': 1.0, 'VRPBK': 1.0,
    'OVRPL': 1.0,  'OVRPTW': 1.0, 'OVRPK': 1.0,
    'VRPLTW': 1.0, 'VRPLK': 1.0,  'VRPTWK': 1.0,
}

env_params = {
    'problem_type': 'CVRP',
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128, 'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6, 'qkv_dim': 16, 'head_num': 8,
    'logit_clipping': 10, 'ff_hidden_dim': 512, 'eval_type': 'argmax',
    'use_icl': True, 'contrast_weight': 1.0,
    'use_trigram': True, 'use_cooccur': True,
    'edge_encoder_hidden': 64, 'ctx_scale_init': 1.0,
    'edge_gate_bias_init': -1.5,
    'edge_gate_hidden': 32,
}

optimizer_params = {
    'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
    'scheduler': {
        'milestones': [450, 650, 800, 950],
        'gamma': 0.3,
    },
}

trainer_params = {
    'use_cuda': USE_CUDA, 'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 1000,
    'train_episodes': 16000,
    'train_batch_size': 32,

    # ★ 单一规模: 只在 50 个节点上训练和测试 (不再混合 20/30/40)
    'mixed_scale': {
        'enable': True, 'train_sizes': [50],
        'test_size': 50, 'sample_strategy': 'uniform',
    },
    'original_model_path': './pretrained/vrp50_checkpoint.pt',

   
    'eval': {'test_episodes': 5000, 'eval_interval': 99999},

    'icl': {
        'positive_k': 8, 'negative_k': 8, 'negative_strategy': 'rank',
        'baseline_beta': 0.0, 'weight_tau': 0.5,
        'gen2_temperature': 0.8, 'late_gen_k_scale': 0.5,
    },
    'icl_warmup_epochs': 100,
    'standalone_training': True, 'standalone_loss_weight': 0.3,

    'aug_eval': {'enable': False, 'aug_factor': 8},

    'gen0_diversity': {
        'temp_start': 1.2, 'temp_end': 1.5,
        'noise_std': 0.03, 'encoder_dropout': 0.08,
    },

    
    'inheritance_vis': {
        'enable': False,         
        'vis_interval': 9999,
        'vis_instances': 0,
        'vis_detail_instances': 0,
        'top_edge_percentile': 0.2,
    },

    
    'cross_problem': {
        'train_types': TRAIN_TYPES,
        'test_types': TEST_TYPES,
        'type_sample_strategy': 'weighted',
        'type_weights': TRAIN_TYPE_WEIGHTS,
    },

    
    'cross_problem_eval': {
        'enable': False,
        'eval_interval': 200,
        'test_episodes': 1000,
        'test_sizes': [50],
    },

    
    'deep_analysis': {
        'enable': False,
        'analysis_interval': 200,
        'analysis_instances': 256,
        'gate_buckets': 3,
        'difficulty_buckets': 3,
    },

    
    'gate_aux': {
        'enable': True,
        'mode': 'mean_only',
        'lambda_mean': 0.1,
        'lambda_contrast': 0.05,
        'tau': 0.02,
        'warmup_epochs': 100,
    },

    
    'mgc_eval': {
        'enable': False,
        'interval': 40,
        'n_instances': 1000,
        'problem_size': 50,
        'problem_type': 'CVRP',
    },

    'logging': {
        'model_save_interval': 50,     
        'img_save_interval': 0,        
    },
    'model_load': {'enable': False},
}

logger_params = {
    'log_file': {
        'desc': f"v93_single50_1000ep_noeval_ckpt50_seed{SEED}",
        'filename': 'run_log'
    }
}


def main():
    if DEBUG_MODE: _set_debug_mode()
    create_logger(**logger_params); _print_config(); _check_model()
    trainer = Trainer(env_params=env_params, model_params=model_params,
                      optimizer_params=optimizer_params, trainer_params=trainer_params)
    copy_all_src(trainer.result_folder)
    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 5; trainer_params['train_episodes'] = 128
    trainer_params['eval']['test_episodes'] = 64; trainer_params['eval']['eval_interval'] = 1
    trainer_params['icl_warmup_epochs'] = 2; trainer_params['aug_eval']['enable'] = False
    trainer_params['cross_problem_eval']['eval_interval'] = 2
    trainer_params['cross_problem_eval']['test_episodes'] = 64
    trainer_params['gate_aux']['warmup_epochs'] = 2


def _check_model():
    logger = logging.getLogger('root')
    p = trainer_params['original_model_path']
    if os.path.exists(p): logger.info(f"✓ Dedicated model: {p}")
    else: logger.warning(f"✗ No dedicated model: {p} → Exp5 disabled (论文不需要, 忽略)")


def _print_config():
    logger = logging.getLogger('root')
    logger.info('='*80)
    logger.info('CROSS-PROBLEM ICL v9.3 — FAST MODEL-ONLY (1000 ep, single-scale 50, no eval)')
    logger.info(f'  Target: 快速拿到 1000 轮模型文件, 无评估/可视化')
    logger.info(f'  Train types ({len(TRAIN_TYPES)}), Test ({len(TEST_TYPES)})')
    logger.info(f'  Epochs: {trainer_params["epochs"]}')
    logger.info(f'  Scheduler: {optimizer_params["scheduler"]["milestones"]}')

    # 评估/可视化状态汇总
    logger.info(f'  --- Evaluation Schedule (本版全部关闭) ---')
    logger.info(f'  eval (basic):           {"OFF (仅末轮跑1次)" if trainer_params["eval"]["eval_interval"] >= trainer_params["epochs"] else "ON interval=" + str(trainer_params["eval"]["eval_interval"])}')
    logger.info(f'  cross_problem_eval:     {"OFF" if not trainer_params["cross_problem_eval"]["enable"] else "ON"}')
    logger.info(f'  inheritance_vis:        {"OFF" if not trainer_params["inheritance_vis"]["enable"] else "ON"}')
    logger.info(f'  deep_analysis:          {"OFF" if not trainer_params["deep_analysis"]["enable"] else "ON"}')
    logger.info(f'  mgc_eval:               {"OFF" if not trainer_params["mgc_eval"]["enable"] else "ON"}')
    logger.info(f'  aug_eval:               {"OFF" if not trainer_params["aug_eval"]["enable"] else "ON"}')
    logger.info(f'  training curves:        {"OFF" if trainer_params["logging"]["img_save_interval"] == 0 else "ON"}')
    logger.info(f'  model save interval:    {trainer_params["logging"]["model_save_interval"]}')

    cp = trainer_params['cross_problem']
    logger.info(f'  --- Type sampling: {cp["type_sample_strategy"]} ---')
    if cp['type_sample_strategy'] == 'weighted':
        tw = cp['type_weights']
        total_w = sum(tw.values())
        sorted_types = sorted(tw.items(), key=lambda x: -x[1])
        for t, w in sorted_types[:7]:
            logger.info(f'    {t:10s}  weight={w:.1f}  prob={w/total_w*100:.2f}%')
        logger.info(f'    ... ({len(tw) - 7} more @ uniform)')

    ga = trainer_params.get('gate_aux', {})
    logger.info(f'  Gate Aux: mode={ga.get("mode")} (与 v2 一致)')
    logger.info(f'  SEED: {SEED}')
    logger.info('='*80)


if __name__ == "__main__":
    main()
