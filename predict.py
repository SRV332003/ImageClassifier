import argparse
from model_utils import imshow, process_image, predict, load_checkpt
import json
import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("image_path")
parser.add_argument("checkpoint")
parser.add_argument("--gpu", action  = "store_true")
parser.add_argument("--top_k", type =int)
parser.add_argument("--category_names")

args = parser.parse_args()

image = process_image(args.image_path)

image = image.resize_(1,3,224,224)

model = load_checkpt(args.checkpoint,args.gpu)

if args.category_names:
    with open(args.category_names, 'r') as f:
            dic = json.load(f)
else:
    dic = model.class_to_idx

if args.top_k:
    p,classes = predict(image,model,args.top_k,gpu = args.gpu)
else:
    p,classes = predict(image,model,gpu = args.gpu)

names  = [dic[str(int(x))] for x in classes[0]]


print(args.image_path)
print()
for idx,name in enumerate(names):
    print("%30s(%2d)   %f"%(name,classes[0][idx],p[0][idx]))

          
print("")
