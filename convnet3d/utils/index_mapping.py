import operator 
import numpy as np


SUPPORTED_LAYERS = {
    'Pooling'           : {'constrains': [('padding', 'valid', operator.eq)]},
    'Conv'              : {'constrians': [('padiing', 'valid', operator.eq)]},
    'Input'             : {},
    'BatchNormalization': {},
    'Activation'        : {}
}


def constrainsToStrs(constrains):
    cst_strs = []
    fmt = '{}{}==True'
    for cst in constrains:
        operator = cst[-1]
        operands = tuple(cst[:-1])
        cst_strs.append(fmt.format(operator.__name__, operands))
    return cst_strs


def _printLayersInfo(model):
    # for debugging
    for idx, layer in enumerate(model.layers):
        print('{:3d} class name:{}, layer config {}'.format(idx, layer.__class__.__name__, layer.get_config()))
        

def configureChecker(config):
    def decoratedOperator(operator):
        def prequeryOperands(*operands):
            keys = config.keys()
            modified_operands = []
            for opd in operands:
                if opd in keys:
                    modified_operands.append(config[opd])
                else:
                    modified_operands.append(opd)
            return operator(*modified_operands)
        return prequeryOperands
    return decoratedOperator
            

class IndexMap:
    def __init__(self, model):
        if (isinstance(model.inputs, list) and len(model.inputs) > 1) or (isinstance(model.outputs, list)and len(model.outputs) > 1):
            raise ValueError('Multi inputs/outputs model is not support.')

        self.D = 1
        self.C = 0
        self._calcParams(model)

#        _printLayersInfo(model)

    def _calcParams(self, model):
        for idx, layer in enumerate(model.layers[::-1]):
            name = layer.__class__.__name__
            config = layer.get_config()
            supported, category = IndexMap.isSupportedLayer(name, config)
            if not supported and not category:
                raise ValueError('Not supported layer {} at {}'.format(name, len(model.layers) - 1 - idx))
            elif not supported:
                raise ValueError('Layer {} at {} with illegal configuration. (Note the constrains: {})'.format(config['name'], len(model.layers) - 1 - idx, constrainsToStrs(SUPPORTED_LAYERS[category]['constrains'])))
            elif category == 'Pooling' or category == 'Conv':
                kernel_size = np.array(config['kernel_size']) if category == 'Conv' else np.array(config['pool_size'])
                strides     = np.array(config['strides'])
                self.D *= strides
                self.C = (kernel_size - 1) // 2  + self.C * strides

    def isSupportedLayer(name, config):
        supported_categories = SUPPORTED_LAYERS.keys()
        layer_category = None
        for cat in supported_categories:
            if cat in name:  # if this layer is included in category cat
                layer_category = cat
                try:
                    constrains = SUPPORTED_LAYERS[cat]['constrains']

                    for cst in constrains:
                        operator = cst[-1]
                        operands = cst[:-1]

                        checker = configureChecker(config)(operator)
                        if not checker(*operands):
                            # unpack arguemnst for checker
                            # violate the constrains
                            return False, layer_category

                    # supported layer
                    return True, layer_category
                except KeyError:
                    # no constrinas, supported layer
                    return True, layer_category
                    
        # Not suppoerted categories
        return False, layer_category

    def __call__(self, indices):
        return self._mapping(indices)

    def _mapping(self, indices):
        def onePointMapping(point):
            return self.D * point + self.C

        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        
        if indices.shape[-1] != self.C.size:
            raise ValueError('indices shape {} isn\'t consistant with model input size {}'.format(indices.shape, self.C.size))
        return np.apply_along_axis(onePointMapping, -1, indices)
        

