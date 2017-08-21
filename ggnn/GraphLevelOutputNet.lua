-- Output network for making graph level predictions.
--
-- Yujia Li, 03/2016
--

local GraphLevelOutputNet, BaseOutputNet = torch.class('ggnn.GraphLevelOutputNet')

-- output_net_sizes: number of units on each layer of the classification net,
--      the first number in the list is the input size, and the last number is 
--      the output size.
function GraphLevelOutputNet:__init(state_dim, annotation_dim, output_net_sizes, module_dict)
    self.output_net_sizes = output_net_sizes
    self.n_classes = output_net_sizes[#output_net_sizes]
    self.class_net_in_dim = output_net_sizes[1]
    self.state_dim = state_dim
    self.annotation_dim = annotation_dim
    self.module_dict = module_dict or {}
    self:create_aggregation_net_modules()
    self:create_output_net()
    self.vgg_net = vg.load_vggnet()
end

-- Return a list of parameters and a list of parameter gradients. This function
-- is inspired by the parameters function in nn/Container.lua
function GraphLevelOutputNet:parameters()
    local w = {}
    local dw = {}

    -- sort the keys to make sure the parameters are always in the same order
    local k_list = {}
    for k, v in pairs(self.module_dict) do
        table.insert(k_list, k)
    end
    table.sort(k_list)

    for i=1,#k_list do
        m = self.module_dict[k_list[i]]
        local mw, mdw = m:parameters()
        if mw then
            if type(mw) == 'table' then
                for i=1,#mw do
                    table.insert(w, mw[i])
                    table.insert(dw, mdw[i])
                end
            else
                table.insert(w, mw)
                table.insert(dw, mdw)
            end
        end
    end
    return w, dw
end

function GraphLevelOutputNet:getParameters()
    local params, grad_params = self:parameters()
    return nn.Module.flatten(params), nn.Module.flatten(grad_params)
end

-- Creates a copy of this network sharing the same module_dict - i.e. using 
-- exactly the same set of parameters.
function GraphLevelOutputNet:create_share_param_copy()
    return ggnn.GraphLevelOutputNet(self.state_dim, self.annotation_dim, self.output_net_sizes, self.module_dict)
end

function GraphLevelOutputNet:get_constructor_param_dict()
    return {
        state_dim=self.state_dim,
        output_net_sizes=self.output_net_sizes
    }
end

function ggnn.load_graph_level_output_net_from_file(file_name)
    local d = torch.load(file_name)
    local net = ggnn.GraphLevelOutputNet(
        d['state_dim'],
        d['output_net_sizes']
    )
    local w = net:getParameters()
    w:copy(d['params'])

    return net
end

-- The input is a concatenation of the final node representations and the initial
-- node annotations.  Then both the gates and the transformed representations
-- get input from the concatenated input.
--
-- Aggregation net creates graph level representations for each graph, these 
-- networks need to be dynamically created for each graph as different graphs
-- have different sizes.
function GraphLevelOutputNet:create_aggregation_net(n_nodes_list)
    local graph_input = nn.Identity()()
    local image_input = nn.Identity()()
    local input = nn.Reshape(#n_nodes_list*state_dim, 1)(graph_input)
    local first_output = ggnn.create_or_share('Linear', ggnn.AGGREGATION_NET_PREFIX .. '-input', self.module_dict, {self.state_dim, self.class_net_in_dim})(input)
    local second_output = self.vgg_net(image_input)
    local feature_output = nn.Concat(1)({first_output, second_output})
    local output = nn.Sigmoid()(feature_output)
    self.aggregation_net = nn.gModule({graph_input, image_input}, {output})
end

function GraphLevelOutputNet:forward(node_representations, n_nodes_list, imagebatch)
    self:create_aggregation_net(n_nodes_list)
    self.aggregation_net_input = node_representations
    return self.aggregation_net:forward(self.aggregation_net_input)
end

function GraphLevelOutputNet:backward(output_grad)
    local input_grad = self.aggregation_net:backward(self.aggregation_net_input, output_grad)
    return input_grad
end

function GraphLevelOutputNet:str_repr()
    return '<GraphLevelOutputNet>'
end
