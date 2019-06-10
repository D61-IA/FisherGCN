import tensorflow as tf
import scipy

from absl import flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout( x, rate ):
    """Dropout for sparse tensors."""

    random_tensor = 1-rate
    random_tensor += tf.random_uniform( tf.shape(x.values) )
    dropout_mask = tf.cast( tf.floor(random_tensor), dtype=tf.bool )
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./(1-rate))

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call( self, inputs, idx=0 ):
        return inputs

    def __call__( self, inputs, idx=0 ):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call( inputs, idx )
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable(
                                          name='weights',
                                          shape=[input_dim, output_dim] )
            if self.bias:
                self.vars['bias'] = tf.get_variable(
                                        "bias",
                                        shape=[output_dim],
                                        initializer=tf.zeros( [output_dim] ) )
        if self.logging:
            self._log_vars()

    def _call( self, inputs, idx=0 ):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout( x, self.dropout )
        else:
            try:
                x = tf.nn.dropout( x, rate=self.dropout )
            except:
                x = tf.nn.dropout( x, keep_prob=1-self.dropout )

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__( self, input_dim, input_rows, output_dim, support, dropout=0.,
                  sparse_inputs=False, act=tf.nn.relu, bias=False,
                  featureless=False, diag_tensor=False, perturbation=None, **kwargs ):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.diag_tensor = diag_tensor
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_rows = input_rows
        self.perturbation = perturbation

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = tf.get_variable(
                                                 'weights_' + str(i),
                                                 shape=[input_dim, output_dim] )

            if self.bias:
                self.vars['bias'] = tf.get_variable(
                                        "bias",
                                        shape=[output_dim],
                                        initializer=tf.zeros( [output_dim] ) )

            if self.diag_tensor:
                self.vars['diag_tensor'] = tf.get_variable(
                                            "diag_tensor",
                                            shape=1,
                                            initializer=tf.zeros( 1 ) )

        if self.logging:
            self._log_vars()

    def _call( self, inputs, idx=0 ):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout( x, self.dropout )
        else:
            try:
                x = tf.nn.dropout( x, rate=self.dropout )
            except:
                x = tf.nn.dropout( x, keep_prob=1-self.dropout )

        # convolve
        supports = list()
        for i in range( len(self.support) ):
            if not self.featureless:
                pre_sup = dot( x, self.vars['weights_' + str(i)],
                               sparse=self.sparse_inputs )
            else:
                pre_sup = self.vars['weights_' + str(i)]

            if self.diag_tensor:
                mupls = tf.tile(self.vars['diag_tensor'], tf.constant([self.input_rows]))
                diagm = tf.matrix_set_diag( tf.eye(self.input_rows), tf.tile(self.vars['diag_tensor'], tf.constant([self.input_rows])) )
                diag_add = tf.sparse_add( self.support[i], diagm )
                support = dot( diag_add, pre_sup, sparse=False )
                supports.append( ( support ) )

            elif self.perturbation is None:
                if FLAGS.model in ( 'gcn', 'gcnT', 'fishergcn', 'fishergcnT' ):
                    support = dot( self.support[i], pre_sup, sparse=True )
                    supports.append( support )

                elif FLAGS.model == 'gcnR':
                    N = self.input_rows
                    indices = tf.random.uniform( [int(N*N*FLAGS.mask_prob),2], 0, N, dtype=tf.int64 )
                    values  = -tf.ones( (int(N*N*FLAGS.mask_prob),), dtype=tf.float32 )
                    cA = tf.math.abs( tf.sparse.add( self.support[i], tf.SparseTensor( indices, values, [N,N] ) ) )
                    cA = tf.sparse.add( cA, tf.sparse.eye( N ) )
                    cA = tf.sparse.add( cA, tf.sparse.transpose(cA) )
                    _inv_degree = tf.pow( tf.sparse.reduce_sum( cA, axis=1, keepdims=True ), -0.5 )

                    pre_sup = pre_sup * _inv_degree
                    support = dot( cA, pre_sup, sparse=True )
                    support = support * _inv_degree
                    supports.append( support )

                else:
                    raise RuntimeError( 'unknown model' )

            else:
                support = dot( self.support[i], pre_sup, sparse=True )

                # make the perturbation
                FisherU, inc_sigma = self.perturbation
                _noise = dot( tf.transpose( FisherU ), pre_sup, sparse=False )
                _noise *= inc_sigma[:,idx:(idx+1)]
                support -= dot( FisherU, _noise, sparse=False )

                supports.append( support )

            # centering
            #  if self.input_dim > 200:
                # _mean, _var = tf.nn.moments( pre_sup, 0 )
                # pre_sup -= _mean
                # pre_sup /= tf.sqrt( _var )

        output = tf.add_n( supports )

        if self.bias: output += self.vars['bias']

        return self.act( output )

