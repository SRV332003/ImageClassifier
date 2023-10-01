import torch
from torch import nn
import torchvision.models as models

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json



def load_checkpt(path,gpu):
    if not(gpu):
        checkpt = torch.load(path, map_location='cpu')
        
    checkpt = torch.load(path)


    if checkpt["arch"] == "vgg":
        model = models.vgg19(pretrained=False)
    elif checkpt["arch"] == "densenet":
        model = models.densenet121(pretrained=False)
        
    model.classifier = nn.Sequential(nn.Linear(checkpt["in_layer"],checkpt["hidden_layer"],True),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5), 
                                     nn.Linear(checkpt["hidden_layer"],checkpt["hidden_layer"],True),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(checkpt["hidden_layer"],checkpt["out_layer"],True),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpt["state_dict"])
    model.class_to_idx = checkpt["class_to_idx"]
    model.lr = checkpt["lr"]
    model.epochs = checkpt["epochs"] 
    return model

def process_image(image_path,ax =False):
    image = Image.open(image_path)
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
    h,w = image.size
    if h<w:
        image = image.resize(((256*w)//h,256))
    else:
        image = image.resize((256,(256*h)//w))

    #taking center crop
    h,w = image.size
    left,top = (w-224)//2,(h-224)//2
    
    img = image.crop((left,top,left+224,top+224))
    
    #converting to numpy array
    np_img = np.array(img)
    
    #normalizing
    np_img = np_img / 255.0
    std_dev = np.array([0.229, 0.224, 0.225])
    avg = np.array([0.485, 0.456, 0.406])
    np_img = (np_img - avg)/ std_dev

    np_img =torch.from_numpy(np_img).type(torch.FloatTensor)

    if ax :
        fig,ax = plt.subplots()
        ax.imshow(np_img)

    #print(np_img.shape)

    #tranposing
    final_img = np_img.permute(2,0,1) 
    # as 2-idx dim need to be at first and other have to retain dimension
    
    return final_img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))


    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    ax.imshow(image)


    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)


    return ax

def predict(image, model, topk=5,gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
    model.eval()
    with torch.no_grad():
        if gpu:
            image.to(torch.device("cuda"))
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_val,top_class = ps.topk(topk)
    model.train()

    return top_val,top_class
