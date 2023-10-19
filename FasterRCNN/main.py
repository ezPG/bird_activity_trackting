import torch
from eval_rcnn import evaluate_faster_rcnn
from engine import train_one_epoch
import utils
import transforms as T
from model import get_model
from dataset import CustomFasterRCNNDataset
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc



def plot_roc_auc(precision, recall):
    # Compute false positive rate and true positive rate from precision and recall
    fpr = 1 - precision
    tpr = recall

    # Sort fpr and tpr values in ascending order
    sorted_indices = fpr.argsort()
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    # Plot ROC-AUC curve
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("roc_auc_curve.png")

    # Calculate area under ROC curve
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset = CustomFasterRCNNDataset('train_data_faster_rcnn/train/images','train_data_faster_rcnn/train/labels', get_transform(train=True))
    dataset_test = CustomFasterRCNNDataset('train_data_faster_rcnn/val/images','train_data_faster_rcnn/val/labels', get_transform(train=False))
    
    print(f"Length of dataset: {len(dataset)}")
    # parse dataset cheack
    # count = 1
    # for i,j in dataset_test:
    #     print(f"Count: {count}")
    #     count+=1
    #     print(i)
    #     print(j)
    # exit()

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 15

    precision_list = []
    recall_list = []
    f1_list = []
    
    # for epoch in tqdm(range(num_epochs)):
    #     # train for one epoch, printing every 10 iterations
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset
    #     precison,recall,f1_score = evaluate_faster_rcnn(model, data_loader_test, device=device)
        
    #     precision_list.append(precison)
    #     recall_list.append(recall)
    #     f1_list.append(f1_score)
    # plot_roc_auc(np.array(precision_list), np.array(recall_list))
    
    # save model
    torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    precison,recall,f1_score = evaluate_faster_rcnn(model, data_loader_test, device=device)
    
    
        

if __name__ == "__main__":  
    main()