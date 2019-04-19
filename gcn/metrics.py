import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.softmax_cross_entropy_with_logits_v2( logits=preds, labels=labels )
    mask = tf.cast( mask, dtype=tf.float32 )
    mask = ( mask / tf.reduce_mean(mask) )
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy( preds, labels, mask ):
    """Accuracy with masking."""
    correct_prediction = tf.cast( tf.equal( tf.argmax(preds, 1), tf.argmax(labels, 1) ), tf.float32 )
    mask = tf.cast( mask, dtype=tf.float32 )
    return tf.reduce_sum( mask * correct_prediction ) / tf.reduce_sum( mask )

    #acc, acc_op = tf.metrics.accuracy( labels=tf.argmax(labels,1), predictions=tf.argmax(preds,1), weights=mask )
    #return acc_op
