--
-- Created by IntelliJ IDEA.
-- User: zsy
-- Date: 8/16/17
-- Time: 3:56 PM
-- To change this template use File | Settings | File Templates.
--
package.path = package.path .. ";./?.lua;./?/init.lua"
require 'ggnn'
require 'rnn'
require 'nn'
require 'nngraph'

color = require 'trepl.colorize'

TOLERANCE = 1e-5
EPS = 1e-8

function print_test_name(test_name)
    print(color.blue(test_name))
end

function create_sample_graph()
    local edges = {{{1,1,2}, {2,1,3}, {1,2,3}}, {{1,1,2}, {1,2,2}, {2,2,1}}}
    local annotations = {{{0,1}, {1,1}, {1,0}}, {{0,0}, {1,0}}}
    local n_edge_types = 2
    local n_total_nodes = #annotations[1] + #annotations[2]

    return edges, annotations, n_edge_types, n_total_nodes
end

print_test_name('[Test BaseGGNN]')
local edges, annotations, n_edge_types, n_total_nodes = create_sample_graph()
local n_steps = 3
local annotation_dim = #annotations[1][1]
local state_dim = 3

local target = torch.randn(n_total_nodes*state_dim, 1)
local c = nn.MSECriterion()

local g = ggnn.BaseGGNN(state_dim, annotation_dim, {}, n_edge_types)
local w, dw = g:getParameters()
dw:zero()

net = nn.Sequential()
net:add(g)

local y = net:forward(edges, n_steps, annotations)
c:forward(y, target)
local dy = c:backward(y, target)
net:backward(dy)

