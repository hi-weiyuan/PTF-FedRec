import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--device', nargs='?', default='cuda:1' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')
    # global argument
    parser.add_argument('--path', nargs='?', default='../../Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of communication round.')
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    parser.add_argument('--use_privacy', type=bool, default=True, help='whether detect privacy.')
    parser.add_argument('--use_swap', type=bool, default=True, help='whether swap items.')
    parser.add_argument('--use_partial', type=bool, default=True, help='whether use noise to protect.')
    parser.add_argument('--privacy_rate', type=float, default=0.1, help='swap rate.')
    parser.add_argument('--select_ratio', type=float, default=0.1, help='pos rate.')
    parser.add_argument('--random_neg_ratio', type=float, default=1., help='pos:neg rate.')



    # server argument
    parser.add_argument('--server_dim', type=int, default=32, help='Dim of server latent vectors.')
    parser.add_argument('--server_layers', nargs='?', default='[32,16]', help="Dim of mlp layers.")
    parser.add_argument('--server_model_type', nargs='?', default="NGCF", help='NCF or LightGCN.')
    parser.add_argument('--global_lr', type=float, default=0.001, help='global Learning rate.')
    parser.add_argument('--data_treat', nargs='?', default="conf", help='conf or pos.')
    parser.add_argument('--server_bs', type=int, default=1024, help='Server Batch size.')
    parser.add_argument('--server_epoch', type=int, default=2, help='Number of server training epochs.')
    parser.add_argument('--transmit_num', type=int, default=30, help='Server transmit item number.')
    parser.add_argument('--lightgcn_layers', type=int, default=3, help='Server lightgcn layers.')


    # client argument
    parser.add_argument('--client_dim', type=int, default=16, help='Dim of client latent vectors.')
    parser.add_argument('--user_layers', nargs='?', default='[32,16]', help="Dim of mlp layers.")
    parser.add_argument('--local_epoch', type=int, default=5, help='Number of local training epochs.')
    parser.add_argument('--local_lr', type=float, default=0.01, help='local Learning rate.')
    parser.add_argument('--local_bs', type=int, default=256, help='Local Batch size.')
    parser.add_argument('--local_pc', type=float, default=1., help='local privacy control.')
    parser.add_argument('--local_model_type', nargs='?', default="NCF", help='NCF or LightGCN.')

    return parser.parse_args()


args = parse_args()
