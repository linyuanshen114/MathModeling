import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
from matplotlib import pyplot as plt
import os,os.path


method='Alexnet'
print('Load model: ',method)
modelAlex = torchvision.models.alexnet(pretrained=True)
print(modelAlex)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#process picture 
def findte(path):
    
    x=default_loader(path)
    x=preprocess(x)
    x = torch.unsqueeze(x, 0)
    start = time.time()
    x = featuresAlex(x)
    x = x.detach().numpy()
    x=np.squeeze(x)
    x=np.ravel(x)
    x/=np.linalg.norm(x)
    return x

#extract features
def featuresAlex(x,model=modelAlex):
    x = model.features(x)
    return x

#similarity calculation
def similarityViaDot(x,y):
    x=np.squeeze(x)
    y=np.squeeze(y)
    x=np.ravel(x)
    y=np.ravel(y)
    x=x/np.linalg.norm(x)
    y=y/np.linalg.norm(y)
    ans=np.dot(x,y)
    return ans


dataFeatures=dict()
for path in os.listdir('./dataset'):
    dataFeatures[path]=findte('./dataset/'+path)#findte函数输入的是一张图片的路径，返回的是这张图片提取特征并处理后的特征的值，
    #这行代码的意思是，给字典里path标签赋值，字典的形式是{“标签1”：值1，“标签2”：“值2......},就是补充了字典冒号后面值的部分
testFeatures=dict()
for path in os.listdir('./test'):
    testFeatures[path]=findte("./test/"+path)
#print(testFeatures)

#target process
def imageMatchViaFeatures(image_path):
    print('Prepare image data!')
    test_image = default_loader(image_path)#加载测试图像
    input_image = preprocess(test_image)
    input_image = torch.unsqueeze(input_image, 0)
    print('Extract features of target image!')
    start = time.time()
    image_feature = featuresAlex(input_image)#提取测试图像特征
    image_feature = image_feature.detach().numpy()
    print(image_feature[0],image_feature.shape)
    print('Time for extracting features: {:.2f}'.format(time.time() - start))
    

    #similarity ordering
    #dataFeatures是一个字典，定义上面的代码写了，字典的形式是{“标签1”：值1，“标签2”：“值2......},
    # for path，feature，path是这个字典里的标签，feature是这个path标签那个图片对应的图像特征
    t=[]
    for path,feature in dataFeatures.items():
        t.append((path,similarityViaDot(image_feature,feature)))#from dataFeatures_findte
        #similarityviadot函数是计算两张图片提取的特征的相似度
        #在这个循环里，把dataFeatures这个字典里每一个元素都拿出来和测试图片计算相似度，然后返回的是（path（标签），相似度）这样一个数据，把50张图片的这个返回值都添加到t这个list里面
    t.sort(key=lambda x:x[1],reverse=True)#给t里面的50个相似度排序

#列表t的形式是[(path1,相似度数值1)，(path2,相似度数值2)......(path50,相似度数值50)],t共有50个形式为（pathi，相似度i）的元素
    for i in range(len(t)):
        print(i,' ',t[i][0],' ',t[i][1])#t[i][0]是t的第i个元素（pathi，相似度i）的第0个元素pathi,t[i][1]是t的第i个元素（pathi，相似度i）的第1个元素相似度i

    #visualization

    totalNum=min(5,len(t))#输出前五
    plt.subplot(3,3,2)
    plt.imshow(plt.imread(image_path))

    for i in range(totalNum):
        plt.subplot(3,3,i+1+3)
        plt.imshow(plt.imread('./dataset/'+t[i][0]))
        plt.xlabel(t[i][1])
        plt.tight_layout()
    if not os.path.exists('./results'):
        os.makedirs('./results')
    plt.savefig(f'./results/features-{"".join(x for x in image_path if x.isalnum())}.jpg',dpi=1500)
    plt.close()
    print('Save features!')
    np.save(f'./results/features-{"".join(x for x in image_path if x.isalnum())}.npy',image_feature)


#test_data circulation
for path in (os.listdir('./test')):
    imageMatchViaFeatures("./test/"+path)