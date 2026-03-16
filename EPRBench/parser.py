
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=24,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="number of epochs to train for")
    # Inference parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Model parameters
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # contrastive learning parameters
    parser.add_argument('--lambda_contrast', type=float, default=0.1, help='Weight for contrastive learning loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive learning')
    
    # Text description toggle
    parser.add_argument('--use_text', '--use-text', dest='use_text', action='store_true', default=True, help='Whether to enable text description')
    parser.add_argument('--no_use_text', '--no-use-text', dest='use_text', action='store_false', help='Disable text description')
    # Text description folder
    parser.add_argument('--text_folder', type=str, default=None, help='Path to text descriptions (default: dataset_folder/scene_descriptions)')
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--dataset_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    
    # Evaluation parameters
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k", 
                        help="Name of the dataset to evaluate on")
    parser.add_argument("--eval_datasets_folder", type=str, default=None, 
                        help="Path to the folder containing the evaluation datasets")
    
    args = parser.parse_args()
    
    if args.dataset_folder == None:
        try:
            args.dataset_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            if args.eval_datasets_folder is None:
                raise Exception("You should set the parameter --dataset_folder or export " +
                                "the DATASETS_FOLDER environment variable as such \n" +
                                "export DATASETS_FOLDER=../datasets_vg/datasets")
            else:
                args.dataset_folder = "dummy_path_for_eval"
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args
