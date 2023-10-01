import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json



class Classifier():
    
    def __init__(self,data_dir):
        self.model = None
        self.data_dir = data_dir
        self.train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                   ])
        
        self.test_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])
                                                  ])
        self.batch_size = 27
        self.epochs = 6
        self.arch = "vgg"
        self.hidden_layer = 4096
        self.out_layer = 102
        self.in_layer = 25088
        self.lr = 0.0003
        
        
    def train(self,epochs = 6,lr = 0.0003,gpu = False,arch = "vgg",hidden_layer = 4096):
        
        if epochs == None:
            epochs = 6
        if lr == None:
            lr = 0.0003
            
        self.epochs = epochs
        self.arch = arch
        self.hidden_layer = hidden_layer
        
        
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'

        # Loading the datasets with ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transforms)
        valid_dataset = datasets.ImageFolder(valid_dir, transform = self.test_transforms)

        # Using the image datasets and the trainforms, to define the dataloaders
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True )
        validloader  = torch.utils.data.DataLoader(valid_dataset, batch_size = self.batch_size)
        
        if arch == "vgg":
            self.in_layer = 25088
            #downloading a pretrained model
            self.model = models.vgg19(pretrained =True)

            # freezing parameters
            for parameter in self.model.parameters():
                parameter.require_grad = False

            # making classifier according to need
            classifier = nn.Sequential(nn.Linear(self.in_layer,hidden_layer,True),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5), 
                                      nn.Linear(hidden_layer,hidden_layer,True),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(hidden_layer,self.out_layer,True),
                                      nn.LogSoftmax(dim=1))
        if arch == "densenet":
            self.in_layer = 1024
            
            self.model = models.densenet121(pretrained=True)
            
             # freezing parameters
            for parameter in self.model.parameters():
                parameter.require_grad = False

            # making classifier according to need
            classifier = nn.Sequential(nn.Linear(self.in_layer,hidden_layer,True),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5), 
                                      nn.Linear(hidden_layer,hidden_layer,True),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(hidden_layer,self.out_layer,True),
                                      nn.LogSoftmax(dim=1))

        # replacing the classifier layers
        self.model.classifier = classifier
        
        # device check & transfer
        if gpu:
            assert torch.cuda.is_available(), "GPU is not availaible, please provide appropriate options"
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        self.model = self.model.to(device)
        
        
        # Training the network 
        
        #Loss criterion
        loss_criterion = nn.NLLLoss()
        
        #Updating objects hyperparam. and initializing optimizer
        self.lr = lr
        optimizer = optim.Adam(self.model.classifier.parameters(),lr = lr)
        
        #Trackers
        run_loss = 0
        count  = 0
        
        #Loop for epochs
        for e in range(epochs):
            print("Epoch",e+1,)
            count = 0
            run_loss =0
            for images,labels in trainloader:
                images,labels = images.to(device),labels.to(device)
                count+=1

                optimizer.zero_grad()

                output = self.model.forward(images)
                loss = loss_criterion(output,labels)
                loss.backward()
                optimizer.step()
                
                #for tracking the percentage of data trained
                print("\r%.2f%s completed"%(count*100/len(trainloader),"%"),end="")
                
                run_loss += loss.item()
            else:
                print("\r",end='')
                validation_loss = 0
                accuracy=0
                self.model.eval()
                vcount=0
                with torch.no_grad():
                    for images,labels in validloader:
                        vcount+=1
                        images,labels = images.to(device),labels.to(device)

                        log_result = self.model.forward(images)
                        validation_loss += loss_criterion(log_result,labels).item()

                        result = torch.exp(log_result)
                        top_out,top_cat= result.topk(1,dim=1)
                        compare = top_cat == labels.view(*top_cat.shape)
                        accuracy += torch.mean(compare.type(torch.FloatTensor))
                    else:
                        print("Train loss", run_loss/len(trainloader))
                        print("Valid Accurracy:",accuracy.item()*100/len(validloader))
                        print("Validation loss", validation_loss/len(validloader))
                self.model.train() 
                pass
            
    
    def test(self,gpu = False):
        test_dir = self.data_dir + '/test'

        #loading test data
        test_dataset = datasets.ImageFolder(test_dir, transform = self.test_transforms)
        testloader  = torch.utils.data.DataLoader(test_dataset, batch_size = self.batch_size)
        
        #Loss criterion
        loss_criterion = nn.NLLLoss()
        
        # device check & transfer
        if gpu:
            assert torch.cuda.is_available(), "GPU is not availaible, please provide appropriate options"
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # validation on the test set
        testing_loss = 0
        self.model.eval()
        vcount=0
        with torch.no_grad():
            accuracy=0
            for images,labels in testloader:
                vcount+=1
                images,labels = images.to(device),labels.to(device)

                log_result = self.model.forward(images)
                testing_loss += loss_criterion(log_result,labels).item()

                result = torch.exp(log_result)
                top_out,top_cat= result.topk(1,dim=1)
                compare = top_cat == labels.view(*top_cat.shape)
                accuracy += torch.mean(compare.type(torch.FloatTensor))
            print("Test ccurracy:",accuracy.item()*100/len(testloader))
            print("Testing loss", testing_loss/len(testloader))
  
    def save_checkpoint(self, destination):
        #loading dictionary
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        # checkpoint creation
        checkpt = {"in_layer":self.in_layer,
                   "out_layer":self.out_layer,
                   "epochs":self.epochs,
                   "lr":self.lr,
                   "arch": self.arch,
                   "hidden_layer":self.hidden_layer,
                   "class_to_idx":cat_to_name,
                   "state_dict": self.model.state_dict(),
                  }

        #ensuring the correct destination path
        assert destination[-4:] == ".pth", "'destination' should be an '.pth' file"

        torch.save(checkpt,destination)

        return None


    