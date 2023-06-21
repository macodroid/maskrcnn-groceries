import torch
from groceries_dataset import GroceriesDataset, collate_fn
from utils.engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
import config_maskrcnn as config
import torchvision

train_groceries_dataset = GroceriesDataset(root_dir=config.train_root_dir, ann_file=config.train_ann_file,
                                           transform='train')
test_groceries_dataset = GroceriesDataset(root_dir=config.test_root_dir, ann_file=config.test_ann_file,
                                          transform='test')

train_dataloader = DataLoader(
    train_groceries_dataset,
    batch_size=config.batch_size,
    shuffle=config.train_shuffle,
    num_workers=config.number_of_workers,
    collate_fn=collate_fn,
)

test_dataloader = DataLoader(
    test_groceries_dataset,
    batch_size=config.batch_size,
    shuffle=config.test_shuffle,
    num_workers=config.number_of_workers,
    collate_fn=collate_fn,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT', progress=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 10)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                          hidden_layer, 10)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.05, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

num_epochs = 20
for e in range(num_epochs):
    train_one_epoch(model, optimizer, train_dataloader, device, num_epochs, print_freq=10)
    lr_scheduler.step()
    evaluate(model, test_dataloader, device=device)

model_scripted = torch.jit.script(model)
model_scripted.save('groceries_maskrcnn.pt')
