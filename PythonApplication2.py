import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import skimage
from skimage import io,filters
from chainer import optimizers,serializers,Variable
import DRAWsolver
import RAMsolver
import net
import netRAM
import data
import matplotlib
import matplotlib.pyplot as plt
import mnist_clutter
import argparse
import visualization
import os.path

def generation():
    mnist = data.load_mnist_data()
    mnist_data = mnist['data'].astype(np.float32)
    labels = mnist['target']

    truncated_mnist_0 = []
    for i in range(70000):
        if labels[i] == 0:
            truncated_mnist_0.append(mnist_data[i])

    truncated_mnist_data = np.asarray(truncated_mnist_0)
    print truncated_mnist_data.shape

    mnist_data /= 255
    #binarize mnist data

    mnist_data = (mnist_data >= 0.5).astype(np.float32)
    mnist_target = mnist['target'].astype(np.int32)

    #test(mnist_data[:4])
    img_size = 784
    n_epoch = 50
    n_glimpse = 50
    batchsize = 100
    n_h = 256
    n_z = 100
    n_read = 4
    n_write = 25

    #load_model = ["model_parameter/all_digits_draw.model_batchsize_200_nepoch_50_nglimpse_64_h_256_z_100_read_2_write_5","model_parameter/all_digits_draw.state_batchsize_200_nepoch_50_nglimpse_64_h_256_z_100_read_2_write_5"]
    load_model = ["model_parameter_20160808/Gen_draw_allDigits_50_valloss_152.731613159.model_batchsize_100_nepoch_50_nglimpse_50_h_256_z_100_read_2_write_5","model_parameter_20160808/Gen_draw_allDigits_50_valloss_152.731613159.state_batchsize_100_nepoch_50_nglimpse_50_h_256_z_100_read_2_write_5"]
    model = net.DRAW(img_size,n_read,n_write,n_h,n_z,28,28,n_glimpse)
    solver = DRAWsolver.DRAWsolver(model,mnist_data,batchsize,n_epoch,use_gpu=True,load_model=load_model)
    #solver.train()
    images = solver.generate_img()
    #images = images.reshape(-1,images.shape[-1])
    print("image shape is {}".format(images.shape))
    for i in range(100):
        visualization.save_imgs(images[i],"generated_img_20160808/img_AllImage_4m50_{}.png".format(i+200))
    
def generate_data(imgsize=100,N=1000):
    #clutter_label = np.load("clutter_mnist_data/ClutteredMnistLabel.npy")
    #d = {}
    #for i in clutter_label:
    #    if i not in d:
    #        d[i] = 0
    #    d[i] += 1
    #print d
    for i in range(4):
        file_str = "clutter_mnist_data/clutter{}_MnistDataReal_{}_{}.npy".format(imgsize,N,i)
        if os.path.isfile(file_str):
            print "file {} already exists".format(file_str)
        else:
            print "begin to generate data set {}".format(i)
            mnist_clutter._sample(idx=i,N=N,imgsize=imgsize,filename="clutter{}".format(imgsize))
        #mnist_clutter.sample(binary=True,idx=i,N=10)
        #mnist_clutter.sample(binary=False,idx=i,N=1000)

def plot_loss():
    import lossPlot
    dirname = [#"cfyRamTrans60MnistModelParam_20160819_epoch_5000_batchsize_500_weightDecay_0.0001_learmrate_0.001",
    #           "cfyAttClMnistTanhModelParam_20160819_epoch_2000_batchsize_500_weightDecay_0.0001_learmrate_0.001",
               #"model_parameter_20160822\cfyAttCl60NewActivation0823epoch20000bs1000wDecay0.0001lr0.001",
               #"model_parameter_20160822\cfyRamMnistoriginal0823epoch1000bs100wDecay0.0005lr0.01_gsize8",
               #"model_parameter_20160822\cfyRamMnist_originalData0823epoch1000bs100wDecay0.0005lr0.01gisze8",
               #"model_parameter_20160822\cfyNewRamMnist0829FromServer",
               #"model_parameter_20160822\cfyAttCl100NewActivationModelParam_20160823_epoch_20000_batchsize_100_weightDecay_0.0001_learmrate_0.001",
               #"model_parameter_20160822\cfyRamMnistNew0901BaseLastTimee1000bs100wDecay0.0005lr0.01gisze8",
               #"model_parameter_20160822\cfyRamMnist0906_LastTime_locdatae1000bs100wDecay0.0005lr0.01gisze8",
               #"model_parameter_20160822\cfyRamMnist0907_EveryTime_locdatae1000bs100wDecay0.0005lr0.01gisze8",
               #"model_parameter_20160822\cfyRamMnist0907_LastTime_locdatae1000bs100wDecay0.0005lr0.01gisze8",
               "model_parameter_20160822/ram_clutter60_lasttime"
               ]
    filename = [#"RamTrans60",
                #"DrawCl60",
                #"DrawCl100_new_1",
                #"RAM_rewardnoback",
                #"RAM_oringal",
                #"RAM_NEW",
                ##"RAM_EveryTime"
                #"RAM_EveryTimeLocData"
                #"RAM_lastTimeLocData"
                "RAM_cl60"
                ]
    for d,f in zip(dirname,filename):
        lossPlot.sample3(dir=d,imgname=f)



def test():
    ax = np.array(np.random.random((5,4)),dtype=np.float32)
    ay = np.array([1,0,1,0,1])
    l1 = L.Linear(4,2)
    opt = optimizers.SGD(lr=0.1)
    opt.setup(l1)
    for epoch in range(100):
        x = Variable(ax)
        y = Variable(ay)
        x1 = F.sigmoid(l1(x))
        l1.zerograds()
        loss = F.softmax_cross_entropy(x1,y)
        loss.backward()
        opt.update()
        if epoch%10 == 0:
            print loss.data



if __name__ == "__main__":
    #plot_loss()
    #generation()
    #generate_data(imgsize=60,N=100000)
