from torchvision.transforms import transforms

train_root_dir = "/home/maco/Documents/photoneo/data/groceries/train"
train_ann_file = "/home/maco/Documents/photoneo/data/groceries/train/.exports/coco-1686932861.coco.json"

test_root_dir = "/home/maco/Documents/photoneo/data/groceries/test"
test_ann_file = "/home/maco/Documents/photoneo/data/groceries/test/.exports/coco-1686932837.coco.json"

batch_size = 2
train_shuffle = True
test_shuffle = False

number_of_workers = 4

train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
])
