import tensorflow as tf

def masked_softmax_cross_entropy( preds, labels, mask ):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.softmax_cross_entropy_with_logits_v2( logits=preds, labels=labels )
    mask = tf.cast( mask, dtype=tf.float32 )

    return tf.reduce_sum( loss * mask ) / tf.reduce_sum( mask )

def masked_accuracy( preds, labels, mask ):
    """Accuracy with masking (in percentage) """

    correct_prediction = tf.cast( tf.equal( tf.argmax(preds, 1), tf.argmax(labels, 1) ), tf.float32 )
    mask = tf.cast( mask, dtype=tf.float32 )

    return 100 * tf.reduce_sum( mask * correct_prediction ) / tf.reduce_sum( mask )
