from models import *
import argparse
import sys
import os
sys.path.insert(0, '..')
from utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='./model_test.pt')
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='./data')

    arguments = parser.parse_args()
    return arguments



# 原始清洁样本分类预测结果
def predict_image(model_path, img_path):
    # args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_model = ResNet18()
    # 目标模型ResNet18
    target_model = target_model.to(device)
    # 加载训练好的模型，入口 args.model_path
    checkpoint = torch.load(model_path)
    from collections import OrderedDict

    try:
        target_model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)
    target_model.eval()

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    img = Image.open(img_path)
    img_ = test_transform(img).unsqueeze(0)

    img_ = img_.to(device)
    output = target_model(img_)
    print('图片的预测分类结果为:', output.max(1)[1].sum().item())
    return output.max(1)[1].sum().item()


# 分类类型
def change_to_class(class_num):
    if class_num == 0:
        name = "飞机"
    elif class_num == 1:
        name = "汽车"
    elif class_num == 2:
        name = "鸟"
    elif class_num == 3:
        name = "猫"
    elif class_num == 4:
        name = "鹿"
    elif class_num == 5:
        name = "狗"
    elif class_num == 6:
        name = "青蛙"
    elif class_num == 7:
        name = "马"
    elif class_num == 8:
        name = "船"
    elif class_num == 9:
        name = "卡车"
    return name

'''
img_path = 'img_AT.png'    # 图片入口img_path
predict = predict_image('FGSM_SDI_model.pth', img_path)
n = change_to_class(predict)
print('图片的预测分类结果为:', n)
'''

# 用FGSM生成对抗样本，输出：图片
def example_gen_fgsm(model_path, img_path):
    epsilon = (8 / 255.) / std
    alpha = (8 / 255.) / std
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_model = ResNet18()
    # 目标模型ResNet18
    target_model = target_model.to(device)
    # 加载训练好的模型，入口 args.model_path
    checkpoint = torch.load(model_path)
    from collections import OrderedDict

    try:
        target_model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)
    target_model.eval()

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    X = Image.open(img_path)
    X = test_transform(X).unsqueeze(0)
    # print(X)
    X = X.to(device)
    img_path = os.path.basename(img_path)
    a = int(img_path[0])
    y = torch.tensor([a], device=device)
    # print(y)
    fgsm_delta = attack_fgsm(target_model, X, y, epsilon, alpha, restarts=1)
    # print(fgsm_delta)
    print(X + fgsm_delta)
    from torchvision.utils import save_image
    save_image(X + fgsm_delta, 'img_AT.png')


# 用PGD生成对抗样本，迭代次数：10，20，50 输出：图片
def example_gen_pgd(model_path, img_path, attack_iters=10):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_model = ResNet18()
    # 目标模型ResNet18
    target_model = target_model.to(device)
    # 加载训练好的模型，入口 args.model_path
    checkpoint = torch.load(model_path)
    from collections import OrderedDict

    try:
        target_model.load_state_dict(checkpoint)
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        target_model.load_state_dict(new_state_dict, False)
    target_model.eval()

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    X = Image.open(img_path)
    X = test_transform(X).unsqueeze(0)
    # print(X)
    X = X.to(device)
    img_path = os.path.basename(img_path)
    a = int(img_path[0])
    y = torch.tensor([a], device=device)
    # print(y)
    pgd_delta = attack_pgd(target_model, X, y, epsilon, alpha, attack_iters, restarts=1)
    # print(pgd_delta)
    print(X + pgd_delta)
    from torchvision.utils import save_image
    save_image(X + pgd_delta, 'img_AT.png')

'''
img_path = '3_1905.jpg'
example_gen_fgsm('FGSM_SDI_model.pth', img_path)
example_gen_pgd('FGSM_SDI_model.pth', img_path, 50)
'''


