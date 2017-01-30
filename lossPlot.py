import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage import io,filters
import re


def plot(train,val=None,title=None,dirname="model_parameter_20160808",filename=None):
    print "length {}".format(len(train))
    plt.title(title)
    plt.xlabel("Iterations");plt.ylabel("train/val loss")

    print "min : {}, max : {}".format(max(train),min(train))

    plt.ylim(ymin=min(train)*1.02,ymax=max(train)*0.98)
    plt.plot(np.arange(len(train)),train,color='g',linestyle='-')
    plt.annotate("Train curve",xy=(32,train[32]),
                 xytext=(32,train[32]+30),
                 arrowprops=dict(facecolor='green'),
                 horizontalalignment='left',
                 verticalalignment='top')

    if val is not None:
        plt.plot(np.arange(len(val)),val,color='r',linestyle='-')
        plt.annotate("Test curve",xy=(4,val[4]),
                     xytext=(4,val[4]+10),
                     arrowprops=dict(facecolor='red'),
                     horizontalalignment='left',
                    verticalalignment='top')
    plt.savefig("{}/{}.png".format(dirname,filename))

def plot_1(train,train_acc=None,val=None,title=None,dirname="model_parameter_20160808",filename=None):
    #print "length {}".format(len(train))
    plt.title(title)
    plt.xlabel("Iterations");plt.ylabel("train/val loss")

    #print "min : {}, max : {}".format(max(train),min(train))

    #plt.ylim(ymin=min(train)*1.02,ymax=max(train)*0.98)
    x = np.concatenate([train,train_acc,val])

    plt.ylim(ymin=min(x),ymax=max(x))
    plt.plot(np.arange(len(train)),train,color='g',linestyle='-')
    plt.annotate("Train loss",xy=(32,train[32]),
                 xytext=(32,train[32]+0.5),
                 arrowprops=dict(facecolor='green'),
                 horizontalalignment='left',
                 verticalalignment='top')


    if train_acc is not None:
        lenth = len(train_acc)
        plt.plot(np.arange(len(train_acc)),train_acc,color='b',linestyle='-')
        plt.annotate("train acc",xy=(lenth//2,train_acc[lenth//2]),
                     xytext=(lenth//2,train_acc[lenth//2]+0.5),
                     arrowprops=dict(facecolor='blue'),
                     horizontalalignment='left',
                    verticalalignment='top')

    if val is not None:
        length = len(val)
        plt.plot(np.arange(len(val)),val,color='r',linestyle='-')
        plt.annotate("val acc",xy=(length-1,val[length-1]),
                     xytext=(length-1,val[length-1]+0.5),
                     arrowprops=dict(facecolor='red'),
                     horizontalalignment='left',
                    verticalalignment='top')
    plt.savefig("{}/{}.png".format(dirname,filename))


def sample1(filename="model_parameter_20160808/Gen_all_digits_train_val_loss_array_batchsize_100_nepoch_50_nglimpse_50_h_256_z_100_read_2_write_5"):
    f = open(filename)
    p = re.compile(r"[0-9]+\.[0-9]+")
    line = f.readline()
    train_loss = []
    val_loss = []
    while line:
        line = line.strip()
        #print line
        if p.search(line):
            train,val = p.findall(line)
            train_loss.append(float(train))
            val_loss.append(float(val))
            #print "tran loss : {}, val loss : {}".format(train,val)
        line = f.readline()
    
    plot(train_loss,val_loss,title="Train loss and val loss",filename="train_val_loss_figure")

def sample2(filename="model_parameter_20160808/train_val_of_original_data.txt"):
    f = open(filename)
    p = re.compile(r"-[0-9]+\.[0-9]+")
    line = f.readline()
    train_loss = []
    while line:
        li = line.strip().split(' ')
        #print line
        if li[0] == "the" and li[1] == "loss":
            if p.search(line):
                train = p.findall(line)[0]
                train_loss.append(float(train))
                #print "tran loss : {}".format(train)
        line = f.readline()
    
    plot(train_loss,title="Train loss",filename="wrong_train_loss")
    
def sample3(dir="cfyAttMnistModelParam_20160810_epoch_500_batchsize_500_weightDecay_0.0001_learmrate_0.001",
            filename="classify_train_val_loss_array",imgname="imgname"):
    f = open("{}/{}".format(dir,filename))
    p = re.compile(r"[-]{0,1}[0-9]+\.[0-9]+")
    line = f.readline()
    train_loss = []
    train_acc = []
    val_acc = []

    while line:
        li = line.strip().split(' ')
        #print line
        if "train" in li:
            if p.search(line):
                train = p.findall(line)[0]
                print train
                train_loss.append(float(train))
                train = p.findall(line)[1]
                train_acc.append(float(train))
                train = p.findall(line)[2]
                val_acc.append(float(train))
                #print "tran loss : {}".format(train)
        line = f.readline()
    
    print "max train accuracy : {}, max validation accuracy: {}".format(np.max(train_acc),np.max(val_acc))
    plot_1(train_loss,train_acc,val_acc,title="Train loss/accuracy, Val Accuracy",
           dirname=dir,
           filename=imgname)

    #plot_1(train_acc,title="Train loss",filename="train_acc")

if __name__ == "__main__":
    filename = "model_parameter_20160808/Gen_all_digits_train_val_loss_array_batchsize_100_nepoch_50_nglimpse_50_h_256_z_100_read_2_write_5"
    sample1(filename)
    filename = "model_parameter_20160808/train_val_of_original_data.txt"
    sample2(filename)