import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/level2_dkt_recsys-level2-recsys-11/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--fe_num",
        default='00',
        type=str,
        help='feature engineering data file path (ex) 00'
    )
    parser.add_argument("--model", default="CATB", type=str, help="model type")
    parser.add_argument("--cat_cv", default=False, type=bool, help="cross validation of catboost")
    parser.add_argument("--n_epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
    parser.add_argument("--l2_leaf_reg", default=3, type=float, help="l2_leaf_reg term")
    parser.add_argument("--verbose", default=100, type=int, help="number of verbose")
    parser.add_argument("--od_pval", default=0.01, type=int, help="catboost's od_pval")
    parser.add_argument("--od_wait", default=100, type=int, help="catboost's od_wait")
    
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--wandb", default=False, type=bool, help="use wandb")
    parser.add_argument("--ratio", default=0.3, type=float, help="test ratio")
    parser.add_argument("--depth", default=8, type=int, help="depth of catboost")


    args = parser.parse_args()

    return args