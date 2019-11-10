# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/husqin/DnCNN-keras
# =============================================================================

# run this to test the model

#Original version

import argparse
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import keras.backend as K

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models','DnCNN_sigma25'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_001.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def fgsm(model, image, y_true, eps=0.1):
   
    y_pred = model.output 
    
    # y_true: 目标真实值的张量。
    # y_pred: 目标预测值的张量。
    loss=K.sum(K.square(y_pred- y_true))/2

    gradient = K.gradients(loss, model.input)
    gradient = gradient[0] 

    adv = image + K.sign(gradient) * eps #fgsm算法
    
    sess = K.get_session() 
    adv = sess.run(adv, feed_dict={ model.input : image}) #注意这里传递参数的情况
    adv = np.clip(adv, 0, 255) #有的像素点会超过255，需要处理
    return adv


# fgsm攻击 函数调用
# 下面部分代码(图像处理)需要根据自己的攻击图像的实际情况进行修改
def fgsm_attack(model,input_img, y_true,epsilons = 1000):

    # 加载准备攻击的模型，对要攻击的图形进行转换
    lpr_model = model 
    img_convert = input_img
    ret_predict = lpr_model.predict(np.array(img_convert)) #进行预测

    #对于y_true既可以采用干净的样本，也可以采用原本网络的输出；原代码采用的是原本网络的输出

    # 计算eps的值
    epsilons = np.linspace(0,1,num=epsilons+1)[1:]

    print("开始使用fgsm进行攻击")
    for eps in epsilons:
        # img_attack = fgsm(lpr_model, img_convert, y_true, eps=eps)
        img_attack = fgsm(lpr_model, img_convert, y_true, eps=eps)

        attack = lpr_model.predict(img_attack)
        # print("预测值：{}".format(attack))

        # 当识别的结果不等时，表示攻击成功
        if np.sum(attack-ret_predict)!=0:
            # print('攻击成功，攻击后的结果为：', attack)
            return img_attack
    return img_attack


if __name__ == '__main__':    
    
    args = parse_args()
    
    
    # =============================================================================
    #     # serialize model to JSON
    #     model_json = model.to_json()
    #     with open("model.json", "w") as json_file:
    #         json_file.write(model_json)
    #     # serialize weights to HDF5
    #     model.save_weights("model.h5")
    #     print("Saved model")
    # =============================================================================

    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # load json and create model
        json_file = open(os.path.join(args.model_dir,'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir,'model.h5'))
        log('load trained model on Train400 dataset by kai')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name),compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        
    for set_cur in args.set_names:  
        
        if not os.path.exists(os.path.join(args.result_dir,set_cur)):
            os.mkdir(os.path.join(args.result_dir,set_cur))
        psnrs = []
        ssims = [] 
        psnrs_adv = []
        ssims_adv = []         
        
        for im in os.listdir(os.path.join(args.set_dir,set_cur)): 
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                #x = np.array(Image.open(os.path.join(args.set_dir,set_cur,im)), dtype='float32') / 255.0
                
                # Get data 
                x = np.array(imread(os.path.join(args.set_dir,set_cur,im)), dtype=np.float32) / 255.0
                np.random.seed(seed=0) # for reproducibility
                y = x + np.random.normal(0, args.sigma/255.0, x.shape) # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_  = to_tensor(y)


                # Clean Process 
                start_time = time.time()
                x_ = model.predict(y_) # inference
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_=from_tensor(x_)
                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y,x_))) # show the image
                    save_result(x_,path=os.path.join(args.result_dir,set_cur,name+'_dncnn'+ext)) # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)


                # Adversarial Attack
                adv_input = fgsm_attack(model=model,input_img=y_,y_true=x)
                adv_input = np.asarray(adv_input).astype(np.float32)
                print("对抗扰动矩阵：{}".format(adv_input))
                added_puturb = np.sum(adv_input-y)
                print("施加的扰动值：{}".format(added_puturb)) 
                adv_input = np.reshape(adv_input,[-1,y.shape[0],y.shape[1],1])
                preds_adv =  model.predict(adv_input)   
                preds_adv = np.reshape(preds_adv,[y.shape[0],y.shape[1]])
                print(x.shape)
                print(preds_adv.shape)                
                psnr_x_adv = compare_psnr(x, preds_adv)
                ssim_x_adv = compare_ssim(x, preds_adv)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    show(np.hstack((y,preds_adv))) # show the image
                    save_result(preds_adv,path=os.path.join(args.result_dir,set_cur,name+'_dncnnAttacked'+ext)) # save the denoised image
                psnrs_adv.append(psnr_x_adv)
                ssims_adv.append(ssim_x_adv)                


    
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        psnr_avg_adv = np.mean(psnrs_adv)
        ssim_avg_adv = np.mean(ssims_adv)
        psnrs_adv.append(psnr_avg_adv)
        ssims_adv.append(ssim_avg_adv)        
        
        if args.save_result:
            save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,set_cur,'results.txt'))
            save_result(np.hstack((psnrs_adv,ssims_adv)),path=os.path.join(args.result_dir,set_cur,'resultsAttack.txt'))
            
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
        log('Attacked results: \n  PSNR = {0:2.2f}dB, SSIM = {1:1.4f}'.format(psnr_avg_adv, ssim_avg_adv))

        
        


