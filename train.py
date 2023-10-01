import argparse
from Classifier import Classifier
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--save_dir", default="")
parser.add_argument("--learning_rate", type = float,default = 0.0003)
parser.add_argument("--epochs",type = int, default = 6)
parser.add_argument("--gpu",action ="store_true")
parser.add_argument("--arch", default = "vgg", choices=["vgg","densenet"])
parser.add_argument("--hidden_layer", default = 4096,type=int)

args = parser.parse_args()

model = Classifier(args.data_dir)

model.train(args.epochs,args.learning_rate,args.gpu,args.arch,args.hidden_layer)

model.test(args.gpu)

model.save_checkpoint(args.save_dir+"checkpoint.pth")