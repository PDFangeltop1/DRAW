import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers,Variable
from chainer import cuda

import numpy as np
import utils


class DRAWencoder_2layers(chainer.Chain):
    def __init__(self,n_in,read_size,n_h,height,width,n_glimpse,use_gpu=None):
        super(DRAWencoder_2layers,self).__init__(
            #Encoder only
            le1=L.LSTM(read_size,n_h),
            le1_0=L.Linear(n_h,n_h),
            le2=L.Linear(n_h,10),
            lattention_0=L.Linear(n_h,n_h),
            lattention_1=L.Linear(n_h,5),
            )
        self.A = width
        self.B = height
        self.n_read = int(np.sqrt(read_size))
        self.read_size = read_size
        self.n_h = n_h
        self.n_glimpse = n_glimpse
        self.use_gpu = use_gpu



    def forward(self,x,h_enc,visualize=None):
        F_att = []
        for i in range(0,self.n_glimpse):
            h_enc_prev = F.identity(h_enc)
            r,Fx,Fy = self.read_attention(x,h_enc_prev)
            if visualize:
                #F_att.append([Fx,Fy])
                F_att.append(cuda.to_cpu(r.data))
            h_enc = self.le1(r)

        labels = self.le2(F.relu(self.le1_0(h_enc)))
        #labels = self.le2(h_enc)

        if visualize is True:
            F_att_ny = np.asarray(F_att)
            #(n_glimpse, batch, imgsize -> batch, n_glimpse, imgsize)
            F_att = F_att_ny.transpose((1,0,2))
        return labels,F_att
    
    def visualize_attention(self,data):
        #x_test shape: not (batchsize,w,h), but (batchsize,w*h)
        #x,c,h_dec = self.get_varialized_data(x_test)
        x, h_enc, t = data
        #print "x.shape is {}, c.shape is {}, h_dec.shape is {}".format(x.data.shape, c.data.shape,h_dec.data.shape)
        labels, F_att = self.forward(x,h_enc,visualize=True)
        return F_att
        
    def __call__(self,data,train=True,visualize=None):
        #x_test shape: not (batchsize,w,h), but (batchsize,w*h)
        x, h_enc, t = data
        #print "x.shape is {}, c.shape is {}, h_dec.shape is {}".format(x.data.shape, c.data.shape,h_dec.data.shape)
        labels, F_att = self.forward(x,h_enc,visualize=visualize)
        if train:
            loss = F.softmax_cross_entropy(labels,t)
        else:
            loss = F.accuracy(labels,t)  
        return loss


    def read_attention(self,x,h_enc):
        batchsize = x.data.shape[0]
        Fx,Fy,gamma = self.create_F(h_enc, self.n_read)
        #Fx: (batchsize,N,A)
        #Fy: (batchsize,N,B)
        def filter_img(img,Fx,Fy,gamma,N):
            #img: (batchsize,B,A)
            Fxt = F.transpose(Fx,axes=(0,2,1))
            img = F.reshape(img,(-1,self.B,self.A))

            #print "Fxt shape is {}, image shape is {}".format(Fxt.data.shape,img.data.shape)
            #tmp1 = F.batch_matmul(img,Fxt)
            #print "tmp1.shape is {}".format(tmp1.data.shape)
            #glimpse = F.batch_matmul(Fy,tmp1)
            #print "glimpse.shape is {}".format(glimpse.data.shape)
            glimpse = F.batch_matmul(Fy,F.batch_matmul(img,Fxt))
            glimpse = F.reshape(glimpse,(-1,N*N))
            return glimpse*F.broadcast_to(gamma,(batchsize,N*N))

        x_filtered = filter_img(x,Fx,Fy,gamma,self.n_read)
        return x_filtered,Fx,Fy

        
    def create_F(self,h_dec,N):
        #returns Fx:(batchsize,N,A),
        #        Fy:(batchsize,N,B),
        #        gamma: (batchsize,1)
        if self.use_gpu:
            xp = cuda.cupy
        else:
            xp = np 
        batchsize = h_dec.data.shape[0]
        att_var = self.lattention_1(F.relu(self.lattention_0(h_dec)))
        (gx_t,gy_t,logsig2,logdelta_t,loggamma) = F.split_axis(att_var,int(att_var.data.shape[1]),axis=1)
        assert gx_t.data.shape == (batchsize,1)

        gx = (self.A+1.)/2. * (gx_t+1.)
        gy = (self.B+1.)/2. * (gy_t+1.)
        _delta = (max(self.B,self.A)-1)/(N-1)*F.exp(logdelta_t) 
        #(batch,1) -> (batch,N,1)
        delta = utils.broadcast_second_axis(_delta,N)
        assert(delta.data.shape == (batchsize,N,1))

        _sig2 = F.exp(logsig2)
        #(batch,1) -> (batch,N,1)
        sig2 = utils.broadcast_second_axis(_sig2,N)
        assert(sig2.data.shape == (batchsize,N,1))

        gamma = F.exp(loggamma)
        
        _coeff = (xp.arange(N)-N/2-0.5).reshape(N,1)
        #(batch,1) -> (batch,N,1)
        coeff = xp.broadcast_to(_coeff,(batchsize,N,1))
        
        muX = utils.broadcast_second_axis(gx,N) + coeff*delta
        muY = utils.broadcast_second_axis(gy,N) + coeff*delta

        Fx = utils.filterbank_matrix(muX,sig2,self.A,self.use_gpu)
        Fy = utils.filterbank_matrix(muY,sig2,self.B,self.use_gpu)
        assert(Fx.data.shape == (batchsize,N,self.A))
        return Fx,Fy,gamma

class DRAWencoder(chainer.Chain):
    def __init__(self,n_in,read_size,n_h,height,width,n_glimpse,use_gpu=None):
        super(DRAWencoder,self).__init__(
            #Encoder only
            le1=L.LSTM(read_size,n_h),
            le2=L.Linear(n_h,10),
            lattention=L.Linear(n_h,5,nobias=True),
            )
        self.A = width
        self.B = height
        self.n_read = int(np.sqrt(read_size))
        self.read_size = read_size
        self.n_h = n_h
        self.n_glimpse = n_glimpse
        self.use_gpu = use_gpu



    def forward(self,x,h_enc,visualize=None):
        F_att = []
        F_axis = []
        for i in range(0,self.n_glimpse):
            h_enc_prev = F.identity(h_enc)
            r,Fx,Fy,axises = self.read_attention(x,h_enc_prev)
            if visualize:
                #F_att.append([Fx,Fy])
                F_att.append(cuda.to_cpu(r.data))
                axises_c = []
                for axis in axises:
                    axises_c.append(cuda.to_cpu(axis))
                F_axis.append(axises_c)
            h_enc = self.le1(r)
        labels = self.le2(h_enc)

        if visualize is True:
            F_att_ny = np.asarray(F_att)
            #(n_glimpse, batch, imgsize -> batch, n_glimpse, imgsize)
            F_att = F_att_ny.transpose((1,0,2))
            
            #F_axis: (n_glimpse,5,batch) -> batch, n_glimpse, 5
            F_axis_ny = np.asarray(F_axis)
            F_axis = F_axis_ny.transpose((2,0,1))

        return labels,F_att,F_axis
    
    def visualize_attention(self,data):
        #x_test shape: not (batchsize,w,h), but (batchsize,w*h)
        #x,c,h_dec = self.get_varialized_data(x_test)
        x, h_enc, t = data
        print "x.shape is {}, c.shape is {}, h_dec.shape is {}".format(x.data.shape, t.data.shape,h_enc.data.shape)
        labels, F_att, F_axis = self.forward(x,h_enc,visualize=True)
        #label (batchsize, )
        labels = cuda.to_cpu(labels.data)
        labels = np.argmax(labels,axis=1)
        return labels, F_att, F_axis
        
    def __call__(self,data,train=True,visualize=None):
        #x_test shape: not (batchsize,w,h), but (batchsize,w*h)
        x, h_enc, t = data
        #print "x.shape is {}, c.shape is {}, h_dec.shape is {}".format(x.data.shape, c.data.shape,h_dec.data.shape)
        labels, _, _ = self.forward(x,h_enc,visualize=visualize)
        if train:
            loss = F.softmax_cross_entropy(labels,t)
        else:
            loss = F.accuracy(labels,t)  
        return loss


    def read_attention(self,x,h_enc):
        batchsize = x.data.shape[0]
        Fx,Fy,gamma,axises = self.create_F(h_enc, self.n_read)
        #Fx: (batchsize,N,A)
        #Fy: (batchsize,N,B)
        def filter_img(img,Fx,Fy,gamma,N):
            #img: (batchsize,B,A)
            Fxt = F.transpose(Fx,axes=(0,2,1))
            img = F.reshape(img,(-1,self.B,self.A))

            #print "Fxt shape is {}, image shape is {}".format(Fxt.data.shape,img.data.shape)
            #tmp1 = F.batch_matmul(img,Fxt)
            #print "tmp1.shape is {}".format(tmp1.data.shape)
            #glimpse = F.batch_matmul(Fy,tmp1)
            #print "glimpse.shape is {}".format(glimpse.data.shape)

            #print "img data shape: {}, Fx shape: {}, Fy shape: {}".format(x.data.shape,Fx.data.shape,Fy.data.shape)
            glimpse = F.batch_matmul(Fy,F.batch_matmul(img,Fxt))
            glimpse = F.reshape(glimpse,(-1,N*N))
            return glimpse*F.broadcast_to(gamma,(batchsize,N*N))
        
        x_filtered = filter_img(x,Fx,Fy,gamma,self.n_read)
        return x_filtered,Fx,Fy,axises

        
    def create_F(self,h_dec,N):
        #returns Fx:(batchsize,N,A),
        #        Fy:(batchsize,N,B),
        #        gamma: (batchsize,1)
        if self.use_gpu:
            xp = cuda.cupy
        else:
            xp = np 
        batchsize = h_dec.data.shape[0]
        att_var = self.lattention(h_dec)
        #att_var = (F.tanh(self.lattention(h_dec))-1.0)/2
        (gx_t,gy_t,logsig2,logdelta_t,loggamma) = F.split_axis(att_var,int(att_var.data.shape[1]),axis=1)
        assert gx_t.data.shape == (batchsize,1)

        gx_t = F.tanh(gx_t)
        gy_t = F.tanh(gy_t)
        logdelta_t = F.clip(logdelta_t,-1000.,0.)
        loggamma = F.clip(loggamma,-1000.,0.)
        logsig2 = F.clip(logsig2,-1000.,5.)

        gx = (self.A+1.)/2. * (gx_t+1.)
        gy = (self.B+1.)/2. * (gy_t+1.)
        _delta = (max(self.B,self.A)-1)/(N-1)*F.exp(logdelta_t) 
        #(batch,1) -> (batch,N,1)
        delta = utils.broadcast_second_axis(_delta,N)
        assert(delta.data.shape == (batchsize,N,1))

        _sig2 = F.exp(logsig2)
        #(batch,1) -> (batch,N,1)
        sig2 = utils.broadcast_second_axis(_sig2,N)
        assert(sig2.data.shape == (batchsize,N,1))

        gamma = F.exp(loggamma)
        
        _coeff = (xp.arange(N)-N/2-0.5).reshape(N,1)
        #(batch,1) -> (batch,N,1)
        coeff = xp.broadcast_to(_coeff,(batchsize,N,1))
        
        muX = utils.broadcast_second_axis(gx,N) + coeff*delta
        muY = utils.broadcast_second_axis(gy,N) + coeff*delta

        Fx = utils.filterbank_matrix(muX,sig2,self.A,self.use_gpu)
        Fy = utils.filterbank_matrix(muY,sig2,self.B,self.use_gpu)

        muXLeftTop = muX.data[:,0,0]
        muYLeftTop = muY.data[:,0,0]
        muXRightButtom = muX.data[:,-1,0]
        muYRightButtom = muY.data[:,-1,0]
        width = F.exp(logsig2*0.5).data[:,0]
        assert(Fx.data.shape == (batchsize,N,self.A))
        #return Fx,Fy,gamma,[muXLeftTop,muYLeftTop,muXRightButtom,muYRightButtom,width,gx_t.data[:,0],gy_t.data[:,0],_delta.data[:,0]]
        return Fx,Fy,gamma,[muXLeftTop,muYLeftTop,muXRightButtom,muYRightButtom,width,gx.data[:,0],gy.data[:,0],_delta.data[:,0]]




class DRAW(chainer.Chain):
    def __init__(self,n_in,read_size,write_size,n_h,n_z,height,width,n_glimpse,use_gpu=None):
        super(DRAW,self).__init__(
                #Encoder
                le1=L.LSTM(2*read_size+n_h,n_h),
                le2_mu=L.Linear(n_h,n_z),
                le2_logsig=L.Linear(n_h,n_z),

                #Decoder
                ld1=L.LSTM(n_z,n_h),
                lattention=L.Linear(n_h,5),
                lwrite=L.Linear(n_h,write_size),
            )

        self.A = width
        self.B = height
        self.n_read = int(np.sqrt(read_size))
        self.n_write = int(np.sqrt(write_size))
        self.read_size = read_size
        self.write_size = write_size
        self.n_h = n_h
        self.n_z = n_z
        self.n_glimpse = n_glimpse
        self.use_gpu = use_gpu

    def __call__(self,x_test):
        #x_test shape: not (batchsize,w,h), but (batchsize,w*h)
        #x,c,h_dec = self.get_varialized_data(x_test)
        x,c,h_dec = x_test
        #print "x.shape is {}, c.shape is {}, h_dec.shape is {}".format(x.data.shape, c.data.shape,h_dec.data.shape)
        outs,mus,logsigs,sigs = self.forward(x,c,h_dec,train=False)
        loss = self.loss_func(outs[-1],x,mus,logsigs,sigs)
        return outs,loss

    def encode(self,r,h_dec_prev):
        return self.le1(F.concat([r,h_dec_prev],axis=1))

    def decode(self,z):
        return self.ld1(z)

    def sample(self,h_enc):
        #return of shape(batchsize,n_z)
        mu = self.le2_mu(h_enc)
        logsig = self.le2_logsig(h_enc)
        z = F.gaussian(mu,2*logsig) #Parameter of F.gaussian is mu,logsig
        return z,mu,logsig, F.exp(logsig)

    #####################################################
    def generate_img(self,x_test):
        x,c,h_enc = x_test
        batchsize = x.data.shape[0]
        if self.use_gpu:
            mu_normal = Variable(cuda.to_gpu(np.zeros((batchsize,self.n_z)).astype(np.float32)))
            logsig_normal = Variable(cuda.to_gpu(np.zeros((batchsize,self.n_z)).astype(np.float32)))
        else:
            mu_normal = Variable(np.zeros((batchsize,self.n_z)).astype(np.float32))
            logsig_normal = Variable(np.zeros((batchsize,self.n_z)).astype(np.float32))
         
        #z = F.gaussian(mu_normal,logsig_normal)
        cs = [0]*(self.n_glimpse+1)
        cs[0] = c
        for i in range(self.n_glimpse):
            z = F.gaussian(mu_normal,logsig_normal)
            c_prev = F.identity(cs[i])
            h_enc = self.decode(z)
            cs[i+1] = c_prev + self.write_attention(h_enc)
        return cs[1:]
    #########################################################
    def forward(self,x,c,h_dec,train=True):
        mus = [0]*self.n_glimpse
        logsigs = [0]*self.n_glimpse
        sigs = [0]*self.n_glimpse
        cs = [0]*(self.n_glimpse+1)
        cs[0] = c

        for i in range(0,self.n_glimpse):
            h_dec_prev = F.identity(h_dec)
            c_prev = F.identity(cs[i])
            x_res = x - F.sigmoid(c_prev)
            r = self.read_attention(x,x_res,h_dec_prev)
            h_enc = self.encode(r,h_dec_prev)

            z, mus[i], logsigs[i], sigs[i] = self.sample(h_enc)
            if train:
                h_dec = self.decode(z)
            else:
                h_dec = self.decode(mus[i])
            cs[i+1] = c_prev + self.write_attention(h_dec)
        return cs[1:],mus,logsigs,sigs

    def loss_func(self,c,x,mus,logsigs,sigs):
        batchsize = x.data.shape[0] # == self.batchsize ?
        def binary_crossentropy(t,o):
            eps = 1e-8
            return -(t*F.log(o+eps)+(1.0-t)*F.log(1.0-o+eps))
        Lx = F.sum(binary_crossentropy(x,F.sigmoid(c)))/batchsize

        kl_terms = 0
        #print "glimpse length :{}".format(len(mus))
        for mu,sig,logsig in zip(mus,sigs,logsigs):
            mu_sq = mu**2
            sig_sq = sig**2
            logsig2 = logsig*2
            #kl_terms += F.sum(0.5*(mu_sq+sig_sq-logsig2)-0.5*self.n_glimpse,axis=1)
            kl_terms += F.sum(0.5*(mu_sq+sig_sq-logsig2)-0.5,axis=1)
        assert (kl_terms.data.shape == (batchsize,))
        Lz = F.sum(kl_terms)/batchsize
        loss = Lx+Lz
        return loss, Lx, Lz

    def read_attention(self,x,x_res,h_dec):
        batchsize = x.data.shape[0]
        Fx,Fy,gamma = self.create_F(h_dec, self.n_read)
        #Fx: (batchsize,N,A)
        #Fy: (batchsize,N,B)
        def filter_img(img,Fx,Fy,gamma,N):
            #img: (batchsize,B,A)
            Fxt = F.transpose(Fx,axes=(0,2,1))
            img = F.reshape(img,(-1,self.B,self.A))

            #print "Fxt shape is {}, image shape is {}".format(Fxt.data.shape,img.data.shape)
            #tmp1 = F.batch_matmul(img,Fxt)
            #print "tmp1.shape is {}".format(tmp1.data.shape)
            #glimpse = F.batch_matmul(Fy,tmp1)
            #print "glimpse.shape is {}".format(glimpse.data.shape)
            glimpse = F.batch_matmul(Fy,F.batch_matmul(img,Fxt))
            glimpse = F.reshape(glimpse,(-1,N*N))
            return glimpse*F.broadcast_to(gamma,(batchsize,N*N))

        x_filtered = filter_img(x,Fx,Fy,gamma,self.n_read)
        x_res_filterd = filter_img(x_res,Fx,Fy,gamma,self.n_read)
        return F.concat([x_filtered,x_res_filterd])
        
    def write_attention(self,h_dec):
        batchsize =  h_dec.data.shape[0]
        w_patch = self.lwrite(h_dec)
        w_patch = F.reshape(w_patch,(-1,self.n_write,self.n_write))
        #w_patch = F.reshape(w_patch,(batchsize,self.n_write,self.n_write))

        Fx,Fy,gamma = self.create_F(h_dec,self.n_write)
        #(batch,1) --> (batch,A*B) --> (batch,B,A)
        tmp = F.broadcast_to(1./gamma,(batchsize,self.B*self.A))
        inv_gamma = F.reshape(tmp,(batchsize,self.B,self.A))
        Fyt = F.transpose(Fy,axes=(0,2,1))
        written = F.batch_matmul(Fyt,F.batch_matmul(w_patch,Fx))*inv_gamma
        #written = F.batch_matmul(F.batch_matmul(Fyt,w_patch),Fx)*gamma
        return F.reshape(written,(batchsize,self.B*self.A))

    def create_F(self,h_dec,N):
        #returns Fx:(batchsize,N,A),
        #        Fy:(batchsize,N,B),
        #        gamma: (batchsize,1)
        if self.use_gpu:
            xp = cuda.cupy
        else:
            xp = np 

        batchsize = h_dec.data.shape[0]
        #print "the batchsize is {}".format(batchsize)
        att_var = self.lattention(h_dec)
        (gx_t,gy_t,logsig2,logdelta_t,loggamma) = F.split_axis(att_var,int(att_var.data.shape[1]),axis=1)
        assert gx_t.data.shape == (batchsize,1)

        gx = (self.A+1.)/2. * (gx_t+1.)
        gy = (self.B+1.)/2. * (gy_t+1.)
        _delta = (max(self.B,self.A)-1)/(N-1)*F.exp(logdelta_t) 
        #(batch,1) -> (batch,N,1)
        delta = utils.broadcast_second_axis(_delta,N)
        assert(delta.data.shape == (batchsize,N,1))

        _sig2 = F.exp(logsig2)
        #(batch,1) -> (batch,N,1)
        sig2 = utils.broadcast_second_axis(_sig2,N)
        assert(sig2.data.shape == (batchsize,N,1))

        gamma = F.exp(loggamma)
        
        _coeff = (xp.arange(N)-N/2-0.5).reshape(N,1)
        #(batch,1) -> (batch,N,1)
        coeff = xp.broadcast_to(_coeff,(batchsize,N,1))
        
        muX = utils.broadcast_second_axis(gx,N) + coeff*delta
        muY = utils.broadcast_second_axis(gy,N) + coeff*delta

        Fx = utils.filterbank_matrix(muX,sig2,self.A,self.use_gpu)
        Fy = utils.filterbank_matrix(muY,sig2,self.B,self.use_gpu)
        assert(Fx.data.shape == (batchsize,N,self.A))
        return Fx,Fy,gamma

