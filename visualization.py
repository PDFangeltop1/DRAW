import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# test
def save_skimage(x,dirname,filename):
    for i,xi in enumerate(x):
        patch_size = int(np.sqrt(xi.shape[0]))
        t = xi.reshape(patch_size,patch_size,1)
        t = np.tile(t,(1,1,3))
        t = t/np.max(t)
        io.imsave("{}/{}_{}.png".format(dirname,filename,i),t)
    
def save_matplot(x,filename,figsize_w,figsize_h, patchsize,whole):
    fig,ax = plt.subplots(figsize_h,figsize_w,figsize=(9,9),dpi=100)
    for i,(ai,xi) in enumerate(zip(ax.flatten()[:-1],x)):
        ai.imshow(xi.reshape(patchsize,patchsize),cmap='Greys') # grey, white
        #ai.imshow(xi.reshape(28,28)) colorful, the lighter , the bigger pixel value
        ai.get_xaxis().set_visible(False) 
        ai.get_yaxis().set_visible(False) 
    ax.flatten()[-1].imshow(whole.reshape(100,100),cmap='Greys')
    ax.flatten()[-1].get_xaxis().set_visible(False) 
    ax.flatten()[-1].get_yaxis().set_visible(False) 
    fig.savefig(filename)    

#For image Generation
def save_imgs(x,filename):
    fig,ax = plt.subplots(5,10,figsize=(9,9),dpi=100)
    for i,(ai,xi) in enumerate(zip(ax.flatten(),x)):
        t = xi.reshape(28,28,1)
        t = np.tile(t,(1,1,3))
        io.imsave('generated_img/data_{}.png'.format(i),t)
        ai.imshow(xi.reshape(28,28),cmap='Greys')
        ai.get_xaxis().set_visible(False) 
        ai.get_yaxis().set_visible(False) 
    fig.savefig(filename) 


#For image classification, draw 
def save_matplot_axis(x,axis,filename,figsize_w,figsize_h, patchsize,whole,label):
    import matplotlib.patches as mpatches
    #http ://matthiaseisen.com/pp/patterns/p0203/
    #https ://github.com/mjhucla/Google_Refexp_toolbox/blob/master/google_refexp_py_lib/common_utils.py
    imgsize = int(np.sqrt(whole.shape[0]))
    fig,ax = plt.subplots(figsize_h,figsize_w,figsize=(9,9),dpi=100)
    plt.title("Prediction {}".format(label))
    print "Begin ------>"
    for i,ai in enumerate(ax.flatten()):
        if i%2 == 0:
            ai.imshow(x[i/2].reshape(patchsize,patchsize),cmap='Greys') # grey, white
            #ai.imshow(xi.reshape(28,28)) colorful, the lighter , the bigger pixel value
            ai.get_xaxis().set_visible(False) 
            ai.get_yaxis().set_visible(False)
        else: 
            ai.imshow(whole.reshape(imgsize,imgsize),cmap='Greys')
            idx = i//2
            
            #print type(axis[idx][7])
            print "Attention : {}".format(i)
            print "left top: (%.3f,%.3f), Right Buttom: (%.3f,%.3f), GxGy: (%.3f,%.3f), Delta: %.3f"%(axis[idx][0],axis[idx][1],axis[idx][2],
                                                                                       axis[idx][3],axis[idx][5],axis[idx][6], axis[idx][7])

            bbox_plot = mpatches.Rectangle((axis[idx][0],axis[idx][1]),axis[idx][2]-axis[idx][0],axis[idx][3]-axis[idx][1],
                                           fill=False,edgecolor='green',linewidth=axis[idx][4])
            ai.add_patch(bbox_plot)
            #ai.annotate("left top: (%.3f,%.3f), Rigth Button: (%.3f, %.3f), G:(%.3f, %.3f)"%(axis[idx][0],axis[idx][1],
            #                                                                                 axis[idx][2],axis[idx][3],
            #                                                                                 axis[idx][5],axis[idx][6]),
            #            xy=(0,0),
            #            xytext=(5,5),
            #            arrowprops=dict(facecolor='green'),
            #            horizontalalignment='left',
            #            verticalalignment='top')
            ai.get_xaxis().set_visible(False) 
            ai.get_yaxis().set_visible(False) 
    fig.savefig(filename)    

#For image classification, ram
def save_matplot_ram(locations,filename,figsize_w,figsize_h,label,test_img):
    import matplotlib.patches as mpatches
    #http ://matthiaseisen.com/pp/patterns/p0203/
    #https ://github.com/mjhucla/Google_Refexp_toolbox/blob/master/google_refexp_py_lib/common_utils.py
    fig,ax = plt.subplots(figsize_h,figsize_w,figsize=(9,9),dpi=100)
    plt.title("Prediction {}".format(label))
    imgsize = int(np.sqrt(test_img.shape[0]))
    img = test_img.reshape(imgsize,imgsize)

    for i,ai in enumerate(ax.flatten()):
        #print "{}\n{}\n\n{}".format(i,locations[i][0][0],locations[i][1][0])
        ai.imshow(img,cmap='Greys')
        ai.get_xaxis().set_visible(False) 
        ai.get_yaxis().set_visible(False)
        bbox_plot = mpatches.Rectangle(((locations[i][0]+1)*0.5*imgsize-4,(locations[i][1]+1)*0.5*imgsize-4),8,8,
                                           fill=False,edgecolor='green',linewidth=1)
        ai.add_patch(bbox_plot)
    fig.savefig(filename)      




def save_matplot_ram1(locations,filename,figsize_w,figsize_h,label,test_img):
    import matplotlib.patches as mpatches
    #http ://matthiaseisen.com/pp/patterns/p0203/
    #https ://github.com/mjhucla/Google_Refexp_toolbox/blob/master/google_refexp_py_lib/common_utils.py
    print "locations shape is {}".format(locations)
    fig,ax = plt.subplots(1,1,figsize=(9,9),dpi=100)
    plt.xlabel("Prediction {}".format(label))
    imgsize = int(np.sqrt(test_img.shape[0]))
    img = test_img.reshape(imgsize,imgsize)
    ax.imshow(img,cmap='Greys')

    def f(x):
        margin = 4
        gsize = 8
        loc = (x+1)*0.5*(imgsize-gsize+1)+margin
        loc = np.clip(loc,margin,imgsize-margin)
        loc = np.floor(loc).astype(np.int32)
        return loc

    for i in range(figsize_h*figsize_w):
        #print "{} axis is : ({},{})".format(i,locations[i][0][0],locations[i][1][0])

        loc_x = f(locations[i][0])
        loc_y = f(locations[i][1])
        print "loc_x, loc_y : ({},{})".format(loc_x,loc_y)
        #bbox_plot = mpatches.Rectangle((loc_x-4,loc_y-4),8,8,fill=False,edgecolor='green',linewidth=1)
        bbox_plot = mpatches.Rectangle(((locations[i][0]+1)*0.5*imgsize-4,(locations[i][1]+1)*0.5*imgsize-4),8,8,
                                           fill=False,edgecolor='green',linewidth=1)
        ax.add_patch(bbox_plot)
    fig.savefig(filename)      

