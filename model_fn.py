import tensorflow as tf
import resnet_model

RESNET_SIZE = 32

def cifar10_model_fn(config, features, labels, mode):
    """Model function for CIFAR-10."""

    
    tf.summary.image('images', features, max_outputs=6)
  
    network = resnet_model.cifar10_resnet_v2_generator(
        RESNET_SIZE, config.data_cfg.class_number)
  
    inputs = tf.reshape(features, [-1, config.data_cfg.image_height, 
                                       config.data_cfg.image_width, 
                                       config.data_cfg.image_channel])
  
    logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
  
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)
  
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
  
    # Add weight decay to the loss.
    loss = cross_entropy + config.train_cfg.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
  
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        _INITIAL_LEARNING_RATE = config.train_cfg.init_lr * config.train_cfg.batch_size / 128
        _BATCHES_PER_EPOCH = config.data_cfg.train_number / config.train_cfg.batch_size
       
        if config.train_cfg.lr_policy == "lr_step":
           
            bound_epochs = range(0, config.train_cfg.train_epochs, config.train_cfg.lr_step.epoch)
            boundaries = [int(_BATCHES_PER_EPOCH * epoch) for epoch in bound_epochs] 

            values = [_INITIAL_LEARNING_RATE * (config.train_cfg.lr_step.alpha ** time)
                for time in range(0, len(bound_epochs) + 1)]

            learning_rate = tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32), boundaries, values)
  
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
  
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=config.train_cfg.momentum)
  
        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None
  
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}
  
    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
  
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

