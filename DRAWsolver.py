import sys
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers,serializers,Variable
from chainer import cuda

import numpy as np
import utils
import copy
import mnist_clutter

class DRAWsolver(object):
    def __init__(self,model,data,batchsize,n_epoch,use_gpu,load_model=None,optimizer=optimizers.Adam()):
        self.data = data
        self.batchsize = batchsize
        self.n_epoch = n_epoch
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.model = model
        self.model_hyper_str = "batchsize_{}_nepoch_{}_nglimpse_{}_h_{}_z_{}_read_{}_write_{}".format(batchsize,n_epoch,
                                                                                                      self.model.n_glimpse,
                                                                                                      self.model.n_h,
                                                                                                      self.model.n_z,
                                                                                                      self.model.n_read,
                                                                                                      self.model.n_write)

        if load_model is not None:
            serializers.load_npz(load_model[0],self.model)

            self.optimizer.setup(self.model)
            serializers.load_npz(load_model[1],self.optimizer)
            print "load model successfully!"
        
        #self.data_train, self.data_test = np.split(data,[6803])
        self.data_train, self.data_test = np.split(data,[69900])
        #print "training data number is : {}, test data number is {}".format(self.data_train.shape[0],self.data_test.shape[0])
        if self.use_gpu is not None:
            self.model.to_gpu()
            self.model.use_gpu = True
        self._reset()

    def get_varialized_data(self,x,train=True):
        if self.use_gpu:
            x = Variable(cuda.to_gpu(x.astype(np.float32)))
            c = Variable(cuda.to_gpu(cuda.cupy.zeros_like(x.data).astype(np.float32)))
            if train is True:
                h_dec = Variable(cuda.to_gpu(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32)))
            else:
                h_dec = Variable(cuda.to_gpu(np.zeros((x.data.shape[0],self.model.n_h)).astype(np.float32)))
        else:
            x = Variable(x.astype(np.float32))
            c = Variable(np.zeros_like(x.data).astype(np.float32))
            if train is True:
                h_dec = Variable(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32))
            else:
                h_dec = Variable(np.zeros((x.data.shape[0],self.model.n_h)).astype(np.float32))
        return x,c,h_dec 

    def _reset(self):
        self.epoch = 0
        self.best_val_loss = 100000000
        self.loss_history = []
        self.val_loss_history = []

    def _step(self):
        indexes = np.random.choice(self.data_train.shape[0],self.batchsize,replace=False)
        x,c,h_dec = self.get_varialized_data(self.data_train[indexes])
        outs,mus,logsigs,sigs = self.model.forward(x,c,h_dec)
        loss,Lx,Lz = self.model.loss_func(outs[-1],x,mus,logsigs,sigs)
        self.model.zerograds()
        loss.backward()
        loss.unchain_backward()

        print "Before update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        self.optimizer.update()
        print "After update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        return loss.data

    def generate_img(self):
        outs = self.model.generate_img(self.get_varialized_data(self.data_test,train=False))
        #outs shape: (n_glimpse,batchsize,imgsize)
        batchsize, img_size = outs[0].data.shape
        n_glimpse = len(outs)
        print "glimpse number is {}".format(n_glimpse)
        arr = np.zeros((batchsize,n_glimpse,img_size))
        for i,out in enumerate(outs):
            arr[:,i,:] = cuda.to_cpu(F.sigmoid(out).data)
            #tmp = cuda.to_cpu(F.sigmoid(out).data)
            #print "{}th shape is {}".format(i,tmp.shape)
        #arr shape: (batchsize,n_glimpse,imgsize)
        return arr[:]

    def train(self):
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

        num_train = self.data_train.shape[0]
        iterations_per_epoch = max(num_train/self.batchsize,1)
        num_iterations = self.n_epoch*iterations_per_epoch

        print "how many training data? {}\n  how many iteration per epoch ? {}\n how many iterations to run {}\n".format(num_train,iterations_per_epoch,num_iterations)
       
        for t in xrange(num_iterations):
            loss = self._step()
            print "iteration : {}, loss: {}".format(t,loss)
            epoch_end = (t+1)%iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
            first_it = (t == 0)
            last_it = (t == num_iterations-1)
            if first_it or last_it or epoch_end:
                outs,val_out = self.model(self.get_varialized_data(self.data_test,train=False))
                val_loss = val_out[0].data
                self.loss_history.append(loss)
                self.val_loss_history.append(val_loss)

                log_str ="(Epoch %d/%d) train loss: %f, val loss: %f"%(
                    self.epoch,self.n_epoch,loss,val_loss)
                print log_str
                f = open("model_parameter/Gen_all_digits_train_val_loss_array_{}".format(self.model_hyper_str),'a')
                f.write(log_str)
                f.write("\n")
                f.close()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                serializers.save_npz("model_parameter/Gen_draw_allDigits_{}_valloss_{}.model_{}".format(self.epoch, val_loss,self.model_hyper_str),self.model)
                serializers.save_npz("model_parameter/Gen_draw_allDigits_{}_valloss_{}.state_{}".format(self.epoch, val_loss,self.model_hyper_str),self.optimizer)

      


class DRAWEncodersolver(object):
    def __init__(self,model,data,batchsize,n_epoch,weightDecay,learning_rate,use_gpu,load_model=None):
        self.data = data
        self.batchsize = batchsize
        self.n_epoch = n_epoch
        self.use_gpu = (use_gpu>=0)
        self.optimizer = optimizers.Adam(alpha=learning_rate)
        self.model = model
        self.weightDecay = weightDecay
        if load_model is not None:
            serializers.load_npz(load_model[0],self.model)

            self.optimizer.setup(self.model)
            serializers.load_npz(load_model[1],self.optimizer)
            print "load model successfully!"

        self.model_hyper_str = "batchsize_{}_nepoch_{}_nglimpse_{}_h_{}_weightDecay_{}_read_{}".format(batchsize,n_epoch,
                                                                                                       self.model.n_glimpse, 
                                                                                                       self.model.n_h,
                                                                                                       self.weightDecay,
                                                                                                       self.model.n_read)
        length = data[0].shape[0]

        label_from = 0
        train_index = []
        for i in range(length):
            if data[1][i] == label_from:
                train_index.append(i)
                label_from += 1
                if label_from > 9:
                    break

        #print train_index
        #self.data_train_X = data[0][np.array(train_index)]
        #print self.data_train_X.shape
        #self.data_test_X = data[0][-10:]
        #self.data_train_Y = data[1][np.array(train_index)]
        #self.data_test_Y = data[1][-10:]
        #print self.data_train_Y
        self.data_train_X, self.data_test_X = np.split(data[0],[int(length*9/10)])
        self.data_train_Y, self.data_test_Y = np.split(data[1],[int(length*9/10)])

        #self.data_train_X, self.data_test_X = np.split(data[0],[int(length*6/7)])
        #self.data_train_Y, self.data_test_Y = np.split(data[1],[int(length*6/7)])
        #print "training label shape is {}".format(self.data_train_Y.shape)
        #print "training data number is : {}, test data number is {}".format(self.data_train.shape[0],self.data_test.shape[0])
        if self.use_gpu :
            chainer.cuda.get_device(use_gpu).use()
            self.model.to_gpu()
            self.model.use_gpu = True
        self._reset()
    

    def get_varialized_data_1(self,x,t,train=True):
        if self.use_gpu:
            x = Variable(cuda.to_gpu(x[:self.batchsize].astype(np.float32)),volatile=not train)
            t = Variable(cuda.to_gpu(t[:self.batchsize].astype(np.int32)),volatile=not train)
            h_dec = Variable(cuda.to_gpu(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32)),volatile=not train)
            
        else:
            x = Variable(x[:self.batchsize].astype(np.float32),volatile=not train)
            t = Variable(t[:self.batchsize].astype(np.int32),volatile= not train)
            h_dec = Variable(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32),volatile=not train)
        return x,h_dec,t 

    def get_varialized_data(self,x,t,train=True):
        if self.use_gpu:
            x = Variable(cuda.to_gpu(x[:self.batchsize].astype(np.float32)))
            t = Variable(cuda.to_gpu(t[:self.batchsize].astype(np.int32)))
            h_dec = Variable(cuda.to_gpu(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32)))
            
        else:
            x = Variable(x[:self.batchsize].astype(np.float32))
            t = Variable(t[:self.batchsize].astype(np.int32))
            h_dec = Variable(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32))
        return x,h_dec,t 

    def _reset(self):
        self.epoch = 0
        self.best_val_loss = 0
        self.loss_history = []
        self.val_loss_history = []


    def _getnorm(self):
        import numpy.linalg as LA
        norm = 0
        array = np.array([0])

        #print "begin"
        for param in self.model.params():
            #print "param shape :{}".format(cuda.to_cpu(param.data).shape)
            #print "param shape :{}".format(cuda.to_cpu(param.data).flatten().shape)
            array = np.concatenate((array,cuda.to_cpu(param.data).flatten()))
        #print "shape is {}".format(array.shape)
        return LA.norm(array)

    def _step(self):
        indexes = np.random.choice(self.data_train_X.shape[0],self.batchsize,replace=False)
        cur_x = self.data_train_X[indexes]
        cur_y = self.data_train_Y[indexes]
        #print "current label shape {}".format(cur_y.shape)
        loss = self.model(self.get_varialized_data(cur_x,cur_y,train=True),
                                           train=True)
        self.model.zerograds()
        loss.backward()
        loss.unchain_backward()

        #print "Before update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        #print "Before update, param normal is {}".format(self._getnorm())
        self.optimizer.update()
        #print "After update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        #print "After update, param normal is {}".format(self._getnorm())
        return loss.data

    def visualize_attention(self,train_data=False):
        if train_data:
            F_label,F_att, F_axis = self.model.visualize_attention(self.get_varialized_data(self.data_train_X,self.data_train_Y,train=False))
        else:
            F_label,F_att, F_axis = self.model.visualize_attention(self.get_varialized_data(self.data_test_X,self.data_test_Y,train=False))

        F_att_c = cuda.to_cpu(F_att)
        F_axis_c = cuda.to_cpu(F_axis)
        label_c = cuda.to_cpu(F_label)
        return label_c,F_att_c, F_axis_c
        

    def train_val(self,data,train=False):
        num_data = data[0].shape[0]
        iterations_per_epoch = max(num_data/self.batchsize,1)
        train_acc = 0
        for t in xrange(iterations_per_epoch):
            #print "the {}th iteration".format(t)
            #model_copy = copy.deepcopy(self.model)
            model_copy = copy.deepcopy(self.model) 
            tmp = model_copy(self.get_varialized_data(data[0][t*self.batchsize:(t+1)*self.batchsize],
                                                      data[1][t*self.batchsize:(t+1)*self.batchsize],train=True),train=False)
            train_acc += tmp.data
        return train_acc/iterations_per_epoch


    def train(self,file_name):
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.weightDecay)) #1e-4, 

        num_train = self.data_train_X.shape[0]
        iterations_per_epoch = max(num_train/self.batchsize,1)
        num_iterations = self.n_epoch*iterations_per_epoch


        f = open("{}/classify_train_val_loss_array".format(file_name),'a')
        f.write(self.model_hyper_str)
        f.write("\n")
        f.close()
        print "how many training data? {}\n  how many iteration per epoch ? {}\n how many iterations to run {}\n".format(num_train,iterations_per_epoch,num_iterations)
        for t in xrange(num_iterations):
            loss = self._step()
            if t%300 == 0:
                print "iteration : {}, loss: {}".format(t,loss)
            epoch_end = (t+1)%iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
            first_it = (t == 0)
            last_it = (t == num_iterations-1)
            if first_it or last_it or epoch_end:
                val_acc = self.train_val([self.data_test_X,self.data_test_Y])
                train_acc = self.train_val([self.data_train_X,self.data_train_Y])
                self.loss_history.append(loss)
                self.val_loss_history.append(val_acc)
                log_str ="(Epoch %d/%d) train loss: %f, train accuracy: %f, val accuracy: %f"%(
                    self.epoch,self.n_epoch,loss,train_acc,val_acc)
                print log_str
                f = open("{}/classify_train_val_loss_array".format(file_name),'a')
                f.write(log_str)
                f.write("\n")
                f.close()
                if val_acc > self.best_val_loss:
                    self.best_val_loss = val_acc
                if val_acc > self.best_val_loss or self.epoch%5 == 0:
                    serializers.save_npz("{}/epoch_{}.model".format(file_name,self.epoch),self.model)
                    serializers.save_npz("{}/epoch_{}.state".format(file_name,self.epoch),self.optimizer)
                



class DRAWOnTheFlyEncodersolver(object):
    def __init__(self,model,data,batchsize,n_epoch,weightDecay,learning_rate,use_gpu,load_model=None,cl_data=100):
        self.data = data
        self.batchsize = batchsize
        self.n_epoch = n_epoch
        self.use_gpu = (use_gpu>=0)
        self.optimizer = optimizers.Adam(alpha=learning_rate)
        self.model = model
        self.weightDecay = weightDecay
        self.mnist_data = data[0]


        self.imgsize = cl_data
        self.cl_piece = 8
        if cl_data == 60:
            self.cl_piece = 4
        
        if load_model is not None:
            serializers.load_npz(load_model[0],self.model)

            self.optimizer.setup(self.model)
            serializers.load_npz(load_model[1],self.optimizer)
            print "load model successfully!"

        self.model_hyper_str = "batchsize_{}_nepoch_{}_nglimpse_{}_h_{}_weightDecay_{}_read_{}".format(batchsize,n_epoch,
                                                                                                       self.model.n_glimpse, 
                                                                                                       self.model.n_h,
                                                                                                       self.weightDecay,
                                                                                                       self.model.n_read)
        length = data[0].shape[0]
        label_from = 0
        train_index = []
        for i in range(length):
            if data[1][i] == label_from:
                train_index.append(i)
                label_from += 1
                if label_from > 9:
                    break


        #print train_index
        #self.data_train_X = data[0][np.array(train_index)]
        #print self.data_train_X.shape
        #self.data_test_X = data[0][-10:]
        #self.data_train_Y = data[1][np.array(train_index)]
        #self.data_test_Y = data[1][-10:]
        #print self.data_train_Y
        #self.data_train_X, self.data_test_X = np.split(data[0],[int(length*9/10)])
        #self.data_train_Y, self.data_test_Y = np.split(data[1],[int(length*9/10)])

        self.data_train_X, self.data_test_X = np.split(data[0],[int(length*6/7)])
        self.data_train_Y, self.data_test_Y = np.split(data[1],[int(length*6/7)])
        #print "training label shape is {}".format(self.data_train_Y.shape)
        #print "training data number is : {}, test data number is {}".format(self.data_train.shape[0],self.data_test.shape[0])
        if self.use_gpu :
            chainer.cuda.get_device(use_gpu).use()
            self.model.to_gpu()
            self.model.use_gpu = True
        self._reset()
    

    def get_varialized_data_1(self,x,t,train=True):
        if self.use_gpu:
            x = Variable(cuda.to_gpu(x[:self.batchsize].astype(np.float32)),volatile=not train)
            t = Variable(cuda.to_gpu(t[:self.batchsize].astype(np.int32)),volatile=not train)
            h_dec = Variable(cuda.to_gpu(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32)),volatile=not train)
            
        else:
            x = Variable(x[:self.batchsize].astype(np.float32),volatile=not train)
            t = Variable(t[:self.batchsize].astype(np.int32),volatile= not train)
            h_dec = Variable(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32),volatile=not train)
        return x,h_dec,t 

    def get_varialized_data(self,x,t,train=True):
        if self.use_gpu:
            x = Variable(cuda.to_gpu(x[:self.batchsize].astype(np.float32)))
            t = Variable(cuda.to_gpu(t[:self.batchsize].astype(np.int32)))
            h_dec = Variable(cuda.to_gpu(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32)))
            
        else:
            x = Variable(x[:self.batchsize].astype(np.float32))
            t = Variable(t[:self.batchsize].astype(np.int32))
            h_dec = Variable(np.zeros((self.batchsize,self.model.n_h)).astype(np.float32))
        return x,h_dec,t 

    def _reset(self):
        self.epoch = 0
        self.best_val_loss = 0
        self.loss_history = []
        self.val_loss_history = []


    def _getnorm(self):
        import numpy.linalg as LA
        norm = 0
        array = np.array([0])

        #print "begin"
        for param in self.model.params():
            #print "param shape :{}".format(cuda.to_cpu(param.data).shape)
            #print "param shape :{}".format(cuda.to_cpu(param.data).flatten().shape)
            array = np.concatenate((array,cuda.to_cpu(param.data).flatten()))
        #print "shape is {}".format(array.shape)
        return LA.norm(array)

    def _step(self):
        indexes = np.random.choice(self.data_train_X.shape[0],self.batchsize,replace=False)
        cur_x = self.data_train_X[indexes]
        cur_x_cl = mnist_clutter.clutter_batch(cur_x,self.mnist_data,imgsize=self.imgsize,cl_piece=self.cl_piece)
        cur_y = self.data_train_Y[indexes]
        #print "current label shape {}".format(cur_y.shape)
        loss = self.model(self.get_varialized_data(cur_x_cl,cur_y,train=True),
                                           train=True)
        self.model.zerograds()
        loss.backward()
        loss.unchain_backward()

        #print "Before update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        #print "Before update, param normal is {}".format(self._getnorm())
        self.optimizer.update()
        #print "After update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        #print "After update, param normal is {}".format(self._getnorm())
        return loss.data

    def visualize_attention(self,train_data=False):
        if train_data:
            cur_x_cl = mnist_clutter.clutter_batch(self.data_train_X[:self.batchsize],
                                                   self.mnist_data,imgsize=self.imgsize,cl_piece=self.cl_piece)
            F_att, F_axis = self.model.visualize_attention(self.get_varialized_data(cur_x_cl,self.data_train_Y,train=False))
        else:
            cur_x_cl = mnist_clutter.clutter_batch(self.data_test_X[:self.batchsize],
                                                   self.mnist_data,imgsize=self.imgsize,cl_piece=self.cl_piece)
            F_att, F_axis = self.model.visualize_attention(self.get_varialized_data(cur_x_cl,self.data_test_Y,train=False))

        F_att_c = cuda.to_cpu(F_att)
        F_axis_c = cuda.to_cpu(F_axis)
        return F_att_c, F_axis_c
        

    def train_val(self,data,train=False):
        num_data = data[0].shape[0]
        iterations_per_epoch = max(num_data/self.batchsize,1)
        train_acc = 0
        for t in xrange(iterations_per_epoch):
            #print "the {}th iteration".format(t)
            #model_copy = copy.deepcopy(self.model)
            model_copy = copy.deepcopy(self.model) 
            x_cl = mnist_clutter.clutter_batch(data[0][t*self.batchsize:(t+1)*self.batchsize],self.mnist_data,
                                               imgsize=self.imgsize,cl_piece=self.cl_piece)
            tmp = model_copy(self.get_varialized_data(x_cl,
                                                      data[1][t*self.batchsize:(t+1)*self.batchsize],train=True),train=False)
            train_acc += tmp.data
        return train_acc/iterations_per_epoch


    def train(self,file_name):
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.weightDecay)) #1e-4, 

        num_train = self.data_train_X.shape[0]
        iterations_per_epoch = max(num_train/self.batchsize,1)
        num_iterations = self.n_epoch*iterations_per_epoch


        f = open("{}/classify_train_val_loss_array".format(file_name),'a')
        f.write(self.model_hyper_str)
        f.write("\n")
        f.close()
        print "how many training data? {}\n  how many iteration per epoch ? {}\n how many iterations to run {}\n".format(num_train,iterations_per_epoch,num_iterations)
        for t in xrange(num_iterations):
            loss = self._step()
            if t%10 == 0:
                print "iteration : {}, loss: {}".format(t,loss)
            epoch_end = (t+1)%iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
            first_it = (t == 0)
            last_it = (t == num_iterations-1)
            if first_it or last_it or epoch_end:
                val_acc = self.train_val([self.data_test_X,self.data_test_Y])
                train_acc = self.train_val([self.data_train_X,self.data_train_Y])
                self.loss_history.append(loss)
                self.val_loss_history.append(val_acc)
                log_str ="(Epoch %d/%d) train loss: %f, train accuracy: %f, val accuracy: %f"%(
                    self.epoch,self.n_epoch,loss,train_acc,val_acc)
                print log_str
                f = open("{}/classify_train_val_loss_array".format(file_name),'a')
                f.write(log_str)
                f.write("\n")
                f.close()
                if val_acc > self.best_val_loss:
                    self.best_val_loss = val_acc
                if val_acc > self.best_val_loss or self.epoch%200 == 0:
                    serializers.save_npz("{}/epoch_{}.model".format(file_name,self.epoch),self.model)
                    serializers.save_npz("{}/epoch_{}.state".format(file_name,self.epoch),self.optimizer)
                