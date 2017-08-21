--
-- Created by IntelliJ IDEA.
-- User: zsy
-- Date: 8/17/17
-- Time: 10:06 AM
-- To change this template use File | Settings | File Templates.
--
local GSNN = torch.class('ggnn.GSNN')

-- propagation_net is a BaseGGNN
function GSNN:__init(propagation_net, graph_level_output_net, n_classes)
    self.propagation_net = propagation_net
    self.graph_level_output_net = graph_level_output_net
    self.n_classes = n_classes
end

-- n_prop_steps is the number of propagation steps for each prediction
function GSNN:forward(edges_list, n_prop_steps, annotations_list, image_batch)
    self.n_prop_steps = n_prop_steps
    local pnet = self.propagation_net
    local n_node_list = pnet.n_node_list
    local glnet = self.graph_level_output_net
    local annotations_list_input = annotations_list
    local nodereps
    for t = 1, n_prop_steps do
        nodereps = pnet:forward(edges_list, 1, annotations_list_input) -- node representations
    end
    self.gl_out = glnet:forward(nodereps, n_node_list, image_batch)
    return self.gl_out
end

function GSNN:predict(edges_list, n_prop_steps, annotations_list, image_batch)
    local output = self:forward(edges_list, n_prop_steps, annotations_list, image_batch)
    return output:gt(output)
end

function GSNN:backward(gl_grad)
    local pnet = self.propagation_net
    local glnet = self.graph_level_output_net
    local gl_r_grad = glnet:backward(gl_grad)
    return pnet:backward(gl_r_grad)
end

function GSNN:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

function GSNN:parameters()
    local w, gw = self.propagation_net:parameters()
    local aw, agw = self.graph_level_output_net:parameters()
    for i=1,#aw do
        table.insert(w, aw[i])
        table.insert(gw, agw[i])
    end
    return w, gw
end

---------- model I/O -----------

function GSNN:get_constructor_param_dict()
    return {
        state_dim        = self.propagation_net.state_dim,
        annotation_dim   = self.propagation_net.annotation_dim,
        prop_net_h_sizes = self.propagation_net.prop_net_h_sizes,
        n_edge_types     = self.propagation_net.n_edge_types,
        gl_output_net_sizes   = self.graph_level_output_net.output_net_sizes
    }
end

