import torch
import torch.nn.functional as F
import json
import os
import tempfile
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.data.utils import load_graphs, save_graphs

from DATASET.data_load import SyntheticDataset, PoolDataset, collate
from model import CaseGNN, early_stopping
from bm25_utils import build_bm25_score_store

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "casegnn-matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from train import forward

from torch.utils.tensorboard import SummaryWriter
import time
import logging

import argparse
parser = argparse.ArgumentParser()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


## model parameters
parser.add_argument("--in_dim", type=int, default=768, help="Input_feature_dimension")
parser.add_argument("--h_dim", type=int, default=768, help="Hidden_feature_dimension")
parser.add_argument("--out_dim", type=int, default=768, help="Output_feature_dimension")
parser.add_argument("--dropout", default=0.01, type=float, help="Dropout for embedding / GNN layer ")       
parser.add_argument("--num_head", default=1, type=int, help="Head number of GNN layer ")                            

## training parameters
parser.add_argument("--epoch", type=int, default=100, help="Training epochs")
parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
parser.add_argument("--wd", default=1e-05, type=float, help="Weight decay if we apply some.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--temp", type=float, default=0.1, help="Temperature for relu")
parser.add_argument("--ran_neg_num", type=int, default=1, help="Random sampled case number")
parser.add_argument("--hard_neg", type=str2bool, nargs="?", const=True, default=False, help="Using bm25_neg or not")
parser.add_argument("--hard_neg_num", type=int, default=1, help="Bm25_neg case number")
parser.add_argument('--disable_early_stop', action='store_true', help="Run all epochs without early stopping")
parser.add_argument('--enable_view_weight_fusion', action='store_true', help="Enable learnable fact/issue view fusion. This is already the simple CaseGNN default.")
parser.add_argument('--enable_bm25_fusion', action='store_true', help="Enable weighted fusion with BM25 scores")
parser.add_argument('--disable_bm25_fusion', action='store_true', help="Deprecated; BM25 fusion is disabled by default for simple CaseGNN")
parser.add_argument("--bm25_train_dir", type=str, default=None, help="Optional override for the training summary directory used to build BM25 scores")
parser.add_argument("--bm25_test_dir", type=str, default=None, help="Optional override for the test summary directory used to build BM25 scores")


## other parameters
parser.add_argument("--data", type=str, default='2017', help="coliee2022 or coliee2023")

args = parser.parse_args()

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')
logging.warning(args)


def resolve_summary_dir(split):
    return f"./PromptCase/task1_{split}_{args.data}/summary_{split}_{args.data}_txt"


def maybe_build_bm25_store(split, override_dir=None):
    if args.disable_bm25_fusion or not args.enable_bm25_fusion:
        logging.warning("BM25 fusion disabled for simple CaseGNN.")
        return None

    summary_dir = override_dir or resolve_summary_dir(split)
    if not os.path.isdir(summary_dir):
        logging.warning("BM25 fusion disabled for %s split because %s does not exist.", split, summary_dir)
        return None

    logging.warning("Building BM25 score store for %s split from %s", split, summary_dir)
    return build_bm25_score_store(summary_dir, summary_dir)

def main():
    log_dir = './CaseGNN_experiments/coliee'+args.data+'_bs'+str(args.batch_size)+'_dp'+str(args.dropout)+'_lr'+str(args.lr)+'_wd'+str(args.wd)+'_t'+str(args.temp)+'_headnum'+str(args.num_head)+'_hardneg'+str(args.hard_neg_num)+'_ranneg'+str(args.ran_neg_num)+'_'+time.strftime("%m%d-%H%M%S")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## model initialization
    model = CaseGNN(args.in_dim, args.h_dim, args.out_dim, dropout=args.dropout, num_head=args.num_head)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ## Dataset initialization
    
    # Train dataset
    train_dataset = SyntheticDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_fact_Synthetic.bin")
    train_graph = train_dataset.graph_list
    train_label = train_dataset.label_list
    train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
    train_dataloader = GraphDataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False, collate_fn=collate)

    train_sumfact_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_fact.bin")
    train_referissue_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"train_issue.bin")
    
    # Test dataset
    ##Inference batch size
    if args.data == '2022':
        inference_bs = 1563
    elif args.data == '2023':
        inference_bs = 1335
    else:
        inference_bs = 2
        
    test_sumfact_dataset = SyntheticDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_fact_Synthetic.bin")

    test_sumfact_graph = test_sumfact_dataset.graph_list
    test_sumfact_sampler = SubsetRandomSampler(torch.arange(len(test_sumfact_graph)))
    test_dataloader = GraphDataLoader(
        test_sumfact_dataset, sampler=test_sumfact_sampler, batch_size=inference_bs, drop_last=False, collate_fn=collate, shuffle=False)

    test_sumfact_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_fact.bin")
    test_referissue_pool_dataset = PoolDataset("./Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+"test_issue.bin")

    ## load train label
    train_labels = {}
    with open('./label/task1_train_labels_'+args.data+'.json', 'r')as f:
        train_labels = json.load(f)
        f.close() 

    with open('./label/hard_neg_top50_train_'+args.data+'.json', 'r')as file:
        bm25_hard_neg_dict = json.load(file)
        file.close() 

    train_bm25_score_store = maybe_build_bm25_store('train', args.bm25_train_dir)

    # ## load test label
    test_labels = {}
    with open('./label/task1_test_labels_'+args.data+'.json', 'r')as f:
        test_labels = json.load(f)
        f.close()    

    test_bm25_score_store = maybe_build_bm25_store('test', args.bm25_test_dir)

    yf_path = './label/test_'+args.data+'_candidate_with_yearfilter.json' 

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))

    highest_ndcg = 0
    con_epoch_num = 0
    for epoch in tqdm(range(args.epoch)):
        print('Epoch:', epoch)
        forward(args.data, model, device, writer, train_dataloader, train_sumfact_pool_dataset, train_referissue_pool_dataset, train_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, args.ran_neg_num, train_flag=True, embedding_saving=False, optimizer=optimizer, bm25_score_store=train_bm25_score_store)
        with torch.no_grad():                      
            ndcg_score_yf = forward(args.data, model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, args.ran_neg_num, train_flag=False, embedding_saving=False, optimizer=optimizer, bm25_score_store=test_bm25_score_store)

        if not args.disable_early_stop:
            stop_para = early_stopping(highest_ndcg, ndcg_score_yf, epoch, con_epoch_num)
            highest_ndcg = stop_para[0]
            if stop_para[1]:
                break
            else:
                con_epoch_num = stop_para[2]
    ##CaseGNN Embedding Saving
    forward(args.data, model, device, writer, train_dataloader, train_sumfact_pool_dataset, train_referissue_pool_dataset, train_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, args.ran_neg_num, train_flag=True, embedding_saving=True, optimizer=optimizer, bm25_score_store=train_bm25_score_store)
    forward(args.data, model, device, writer, test_dataloader, test_sumfact_pool_dataset, test_referissue_pool_dataset, test_labels, yf_path, epoch, args.temp, bm25_hard_neg_dict, args.hard_neg, args.hard_neg_num, args.ran_neg_num, train_flag=False, embedding_saving=True, optimizer=optimizer, bm25_score_store=test_bm25_score_store)

if __name__ == '__main__':
    main()
