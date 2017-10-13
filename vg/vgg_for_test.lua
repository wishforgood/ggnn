--
-- Created by IntelliJ IDEA.
-- User: zsy
-- Date: 8/21/17
-- Time: 11:18 AM
-- To change this template use File | Settings | File Templates.
--
package.path = package.path .. ';./init.lua;?/init.lua'
--require 'vg'

local vgg_for_test, Parent = torch.class('vg.vgg_for_test', 'nn.Module')

function vgg_for_test:__init()
    Parent.__init(self)
    self.model = loadcaffe.load('deploy.prototxt', 'VGG-16.caffemodel', cudnn)
end

function vgg_for_test:updateOutput(input)
    return self.model:updateOutput(input)
end

function vgg_for_test:updateGradInput(input, gradOutput)

end

function vgg_for_test:accGradParameters(input, gradOutput)

end

model = vgg_for_test()