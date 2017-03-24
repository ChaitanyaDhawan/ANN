require 'torch'
require 'optim'
require 'nn'

logger = optim.Logger('loss_log.txt')


data = torch.Tensor{
	    {-1,-1,-1,54.35},  
            {1,-1,-1,39.71},
            {-1,1,-1,40.97},
            --{1,1,-1,39.29},
            {-1,-1,1,52.81},
            {1,-1,1,34.99},
            {-1,1,1,43.54},
            {1,1,1,19.20},
            {-1.68,0,0,37.44},
            --{1.68,0,0,31.04},
            {0,-1.68,0,40.44},
            {0,1.68,0,23.33},
            {0,0,-1.68,36.23},
            {0,0,1.68,34.69},
            {0,0,0,55.65}
}



model = nn.Sequential()                 
ninputs = 3; noutputs = 1

model:add(nn.Linear(3,8))
model:add(nn.Sigmoid())
model:add(nn.Tanh())
model:add(nn.Dropout(0.2))
model:add(nn.Linear(8,1))



criterion = nn.MSECriterion()

x, dl_dx = model:getParameters()


feval = function(x_new)
   
   if x ~= x_new then
      x:copy(x_new)
   end

  
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ {4} }]      
   local inputs = sample[{ {1,3} }]   

   dl_dx:zero()

   
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))

   
   return loss_x, dl_dx
end


sgd_params = {
   learningRate = (1e-1),
   learningRateDecay = 1e-2,
  -- weightDecay = 1e-1,
 --  momentum = 1e-1
}


prev2=100
prev=0
for i = 1,500 do
   
   
   current_loss = 0

  
   for i = 1,(#data)[1] do
    
      _,fs = optim.sgd(feval,x,sgd_params)

      current_loss = current_loss + fs[1]
   end

   
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)
   
   logger:add{['training error'] = current_loss}
   logger:style{['training error'] = '-'}
   logger:plot()  
 
end


text = {54.35,
             39.71,
             40.97,
             52.81,34.99,43.54,
              19.20,
              37.44,40.44,
              23.33,36.23,34.69,55.65}

print('id  approx   text')
for i = 1,(#data)[1] do
   local myPrediction = model:forward(data[i][{{1,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end

text2={ 39.29,31.04}
data2=torch.Tensor{{1,1,-1},{1.68,0,0}}
for i = 1,(#data2)[1] do
   local myPrediction = model:forward(data[i][{{1,3}}])
   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1],text2[i]))
end
