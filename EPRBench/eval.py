
import os
import torch
import parser
import logging
logging.getLogger('PIL').setLevel(logging.ERROR)
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

args.features_dim = 14*768
if args.eval_dataset_name.startswith("pitts"):     # set infer_batch_size = 8 for pitts30k/pitts250k
    args.infer_batch_size = args.infer_batch_size // 2
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

from backbone.text_encoder_clip import CLIPTextEncoder

######################################### MODEL #########################################
model = network.SGVPRNet()
model = model.to(args.device)

text_encoder = None
if args.use_text:
    text_encoder = CLIPTextEncoder(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        output_dim=768,
        freeze_clip=True,
    ).to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
    
    # Load text encoder state if available (though CLIP is usually frozen/pretrained)
    checkpoint = torch.load(args.resume, map_location=args.device)
    if isinstance(checkpoint, dict) and "text_encoder_state_dict" in checkpoint and text_encoder is not None:
        text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"], strict=False)


# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
if args.eval_dataset_name == "EventVPR":
    from dataloaders.EventVPRDataset import get_EventVPR
    # dataset_folder is already the full path if eval_datasets_folder points to .../OpenEventVPR and name is EventVPR
    # eval_datasets_folder is /.../OpenEventVPR, name is EventVPR
    # so we want /.../OpenEventVPR/EventVPR
    dataset_folder = join(args.eval_datasets_folder, args.eval_dataset_name)
    test_ds = get_EventVPR(dataset_folder, split="test", use_text=args.use_text, text_folder=args.text_folder)
    # 为兼容 test.py 中的 test_method 逻辑，手动添加 resize 属性
    test_ds.resize = args.resize
    # 为兼容 test.py 中的 dataset_name 属性
    test_ds.dataset_name = args.eval_dataset_name
    # 为兼容 test.py 中 test_method 的设置 (虽然 EventVPRDataset 不使用它，但 test.py 会尝试设置它)
    test_ds.test_method = args.test_method
else:
    test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca, text_encoder=text_encoder)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
