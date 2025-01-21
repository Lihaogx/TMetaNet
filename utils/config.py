import logging
import os
from yacs.config import CfgNode as CN
import shutil
# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Set print destination: stdout / file
    cfg.print = 'both'

    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'auto'

    # Output directory
    cfg.out_dir = 'results'

    # Config destination (in OUT_DIR)
    cfg.cfg_dest = 'config.yaml'

    # Random seed
    cfg.seed = 1

    # Print rounding
    cfg.round = 4

    # Tensorboard support for each run
    cfg.tensorboard_each_run = False

    # Tensorboard support for aggregated results
    cfg.tensorboard_agg = True

    # Additional num of worker for data loading
    cfg.num_workers = 0

    # Max threads used by PyTorch
    cfg.num_threads = 6

    # The metric for selecting the best epoch for each run
    cfg.metric_best = 'auto'

    # If visualize embedding.
    cfg.view_emb = False

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = 'bitcoin-alpha'

    # Task: node, edge, graph, link_pred
    cfg.dataset.task = 'link_pred'

    # Type of task: classification, regression, classification_binary
    # classification_multi
    cfg.dataset.task_type = 'classification'

    
    cfg.dataset.task_splitting = 'within'
    
    # Split ratio of dataset. Len=2: Train, Val. Len=3: Train, Val, Test
    cfg.dataset.split = [0.8, 0.1, 0.1]

    # Whether to use an encoder for the edge features
    cfg.dataset.edge_encoder = True

    # If add batchnorm after edge encoder
    cfg.dataset.edge_encoder_bn = True

    # Dimension for edge feature. Updated by the real dim of the dataset
    cfg.dataset.edge_dim = 128
    
    cfg.dataset.node_dim = 1

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()

    # Training (and validation) pipeline mode
    cfg.train.mode = 'live_update'

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True
    
    cfg.train.stop_live_update_after = 9999999

    cfg.train.internal_validation_tolerance = 5
    
    cfg.train.memory_steps = 5
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    # Model type to use
    cfg.model.type = 'roland'

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'cross_entropy'

    # size average for loss function
    cfg.model.size_average = True

    # Threshold for binary classification
    cfg.model.thresh = 0.5

    # ============== Link/edge tasks only
    # Edge decoding methods.
    #   - dot: compute dot(u, v) to predict link (binary)
    #   - cosine_similarity: use cosine similarity (u, v) to predict link (
    #   binary)
    #   - concat: use u||v followed by an nn.Linear to obtain edge embedding
    #   (multi-class)
    cfg.model.edge_decoding = 'concat'


    # ------------------------------------------------------------------------ #
    # GNN options
    # ------------------------------------------------------------------------ #
    cfg.gnn = CN()

    # Number of layers for message passing
    cfg.gnn.layers_mp = 2

    # Hidden layer dim. Automatically set if train.auto_match = True
    cfg.gnn.dim_inner = 64

    # Type of graph conv: generalconv, gcnconv, sageconv, gatconv, ...
    cfg.gnn.layer_type = 'residual_edge_conv'


    # Whether use batch norm
    cfg.gnn.batchnorm = True

    cfg.gnn.dim_out = 64
    
    cfg.gnn.hidden_dim = 256
    # Activation
    cfg.gnn.act = 'prelu'

    # Dropout
    cfg.gnn.dropout = 0.0

    # Aggregation type: add, mean, max
    # Note: only for certain layers that explicitly set aggregation type
    # e.g., when cfg.gnn.layer_type = 'generalconv'
    cfg.gnn.agg = 'add'

    # Normalize adj
    cfg.gnn.normalize_adj = False

    # Message direction: single, both
    cfg.gnn.msg_direction = 'single'

    # Number of attention heads
    cfg.gnn.att_heads = 1
    
    # Normalize after message passing
    cfg.gnn.l2norm = True

    cfg.gnn.skip_connection = 'affine'

    # ------------------------------------------------------------------------ #
    # Optimizer options
    # ------------------------------------------------------------------------ #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01
    
    cfg.optim.meta_lr = 0.01

    cfg.optim.meta_weight_decay = 5e-2
    
    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9
    
    cfg.optim.meta_loop = 10

    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 100

    # ------------------------------------------------------------------------ #
    # Batch norm options
    # ------------------------------------------------------------------------ #
    cfg.bn = CN()

    # BN epsilon
    cfg.bn.eps = 1e-5

    # BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
    cfg.bn.mom = 0.1

    # ------------------------------------------------------------------------ #
    # Memory options
    # ------------------------------------------------------------------------ #
    cfg.mem = CN()

    # Perform ReLU inplace
    cfg.mem.inplace = False
    
    cfg.experimental = CN()

    # How many negative edges for each node to compute rank-based evaluation
    # metrics such as MRR and recall at K.
    # E.g., if multiplier = 1000 and a node has 3 positive edges, then we
    # compute the MRR using 1000 randomly generated negative edges
    # + 3 existing positive edges.
    cfg.experimental.rank_eval_multiplier = 1000
    
    
    cfg.transaction = CN()

    # whether use snapshot
    cfg.transaction.snapshot = False

    # snapshot split method 1: number of snapshots
    # split dataset into fixed number of snapshots.
    cfg.transaction.snapshot_num = 100

    # snapshot split method 2: snapshot frequency
    # e.g., one snapshot contains transactions within 1 day.
    cfg.transaction.snapshot_freq = 'D'

    cfg.transaction.check_snapshot = False

    # how to use transaction history
    # full or rolling
    cfg.transaction.history = 'full'


    # type of loss: supervised / meta
    cfg.transaction.loss = 'meta'

    # feature dim for int edge features
    cfg.transaction.feature_int_dim = 32
    cfg.transaction.feature_node_int_num = [0]

    # feature dim for amount (float) edge feature
    cfg.transaction.feature_amount_dim = 64

    # feature dim for time (float) edge feature
    cfg.transaction.feature_time_dim = 64

    #
    cfg.transaction.node_feature = 'raw'

    # how many days look into the future
    cfg.transaction.horizon = 1

    # prediction mode for the task; 'before' or 'after'
    cfg.transaction.pred_mode = 'before'

    # number of periods to be captured.
    # set to a list of integers if wish to use pre-defined periodicity.
    # e.g., [1,7,28,31,...] etc.
    cfg.transaction.time_enc_periods = [1]

    # if 'enc_before_diff': attention weight = diff(enc(t1), enc(t2))
    # if 'diff_before_enc': attention weight = enc(t1 - t2)
    cfg.transaction.time_enc_mode = 'enc_before_diff'

    # how to compute the keep ratio while updating the recurrent GNN.
    # the update ratio (for each node) is a function of its degree in [0, t)
    # and its degree in snapshot t.
    cfg.transaction.keep_ratio = 'linear'

    cfg.metric = CN()
    # how to compute MRR.
    # available: f = 'min', 'max', 'mean'.
    # Step 1: get the p* = f(scores of positive edges)
    # Step 2: compute the rank r of p* among all negative edges.
    # Step 3: RR = 1 / rank.
    # Step 4: average over all users.
    # expected MRR(min) <= MRR(mean) <= MRR(max).
    cfg.metric.mrr_method = 'max'

    # Specs for the link prediction task using BSI dataset.
    # All units are days.
    cfg.link_pred_spec = CN()

    # The period of `today`'s increase: how often the system is making forecast.
    # E.g., when = 1,
    # the system forecasts transactions in upcoming 7 days for everyday.
    # One training epoch loops over
    # {Jan-1-2020, Jan-2-2020, Jan-3-2020..., Dec-31-2020}
    # When = 7, the system makes prediction every week.
    # E.g., the system forecasts transactions in upcoming 7 days
    # on every Monday.
    cfg.link_pred_spec.forecast_frequency = 1

    # How many days into the future the model is trained to predict.
    # The model forecasts transactions in (today, today + forecast_horizon].
    # NOTE: forecast_horizon should >= forecast_frequency to cover all days.
    cfg.link_pred_spec.forecast_horizon = 7

        # For meta-learning.
    cfg.roland = CN()
    # Whether to do meta-learning via initialization moving average.
    # Default to False.
    cfg.roland.is_meta = True

    # choose between 'moving_average' and 'online_mean'
    cfg.roland.method = 'moving_average'
    # For online mean:
    # new_mean = (n-1)/n * old_mean + 1/n * new_value.
    # where *_mean corresponds to W_init.

    # Weight used in moving average for model parameters.
    # After fine-tuning the model in period t and get model M[t],
    # Set W_init = (1-alpha) * W_init + alpha * M[t].
    # For the next period, use W_init as the initialization for fine-tune
    # Set cfg.meta.alpha = 1.0 to recover the original algorithm.
    cfg.roland.alpha = 0.8

    
    cfg.windows = CN()
    
    cfg.windows.window_size = 7
    
    cfg.windows.maml_lr = 0.006
    
    cfg.windows.drop_rate = 0.4
    
    cfg.windows.beta = 0.5
    
    
    cfg.topo = CN()
    
    cfg.topo.use_topo = False
    # pixel resolution
    cfg.topo.resolution = 50
    
    cfg.topo.filtration = [[1,1]]
    
    cfg.topo.remove_edge = 'kshell'
        
    cfg.topo.remove_ratio = 1.0
    
    cfg.topo.window_size = 20
    
    cfg.topo.dropout = 0.2
    
    cfg.topo.bandwidth = 1.0
    
    cfg.topo.power = 1.0
    
    cfg.topo.is_directed = False
    
    cfg.topo.distance = 'wasserstein'
    
    cfg.topo.gamma = 0.1
    
    cfg.topo.weight_method = 'exp'
    
    cfg.topo.delta = 0.0
    
    cfg.topo.meta_type = 'MultiChannel'
    
    cfg.topo.drop_rate = 0.1
    
    
def assert_cfg(cfg):
    """Checks config values invariants."""
    if cfg.dataset.task not in ['node', 'edge', 'graph', 'link_pred']:
        raise ValueError('Task {} not supported, must be one of'
                         'node, edge, graph, link_pred'.format(
            cfg.dataset.task))
    if 'classification' in cfg.dataset.task_type and cfg.model.loss_fun == \
            'mse':
        cfg.model.loss_fun = 'cross_entropy'
        logging.warning(
            'model.loss_fun changed to cross_entropy for classification.')
    if cfg.dataset.task_type == 'regression' and cfg.model.loss_fun == \
            'cross_entropy':
        cfg.model.loss_fun = 'mse'
        logging.warning('model.loss_fun changed to mse for regression.')
    if cfg.dataset.task == 'graph' and cfg.dataset.transductive:
        cfg.dataset.transductive = False
        logging.warning('dataset.transductive changed to False for graph task.')
    if cfg.gnn.layers_post_mp < 1:
        cfg.gnn.layers_post_mp = 1
        logging.warning('Layers after message passing should be >=1')


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def update_out_dir(out_dir, fname):
    fname = fname.split('/')[-1][:-5]
    cfg.out_dir = os.path.join(out_dir, fname, str(cfg.seed))
    makedirs_rm_exist(cfg.out_dir)


def get_parent_dir(out_dir, fname):
    fname = fname.split('/')[-1][:-5]
    return os.path.join(out_dir, fname)


def rm_parent_dir(out_dir, fname):
    fname = fname.split('/')[-1][:-5]
    makedirs_rm_exist(os.path.join(out_dir, fname))

def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    
set_cfg(cfg)