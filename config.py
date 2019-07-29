import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='Train', help='Train / Dev / Test')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--num_keywords', type=int, default=96894)
    parser.add_argument('--num_readers', type=int, default=310759)
    parser.add_argument('--num_writers', type=int, default=19066)
    parser.add_argument('--num_items', type=int, default=643105)
    parser.add_argument('--num_magazines', type=int, default=28130)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--dataset_root', type=str,
                        default='/data/private/Arena/datasets/')
    parser.add_argument('--prepro_root', type=str,
                        default='/data/private/Arena/prepro_results/')
    parser.add_argument('--train_dataset_path', type=str,
                        default='train_dataset.pkl')
    parser.add_argument('--valid_dataset_path', type=str,
                        default='valid_dataset.pkl')
    parser.add_argument('--dev_dataset_path', type=str,
                        default='dev_dataset.pkl')
    parser.add_argument('--test_dataset_path', type=str,
                        default='test_dataset.pkl')
    parser.add_argument('--save_path', type=str,
                        default='./models/')

    args = parser.parse_args()
    print(args)

    return args
