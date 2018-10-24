from __future__ import print_function

import os
import re
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
from tensorflow.contrib import slim

flags = tf.app.flags
flags.DEFINE_string("data_dir", "data",
                    "Directory for storing data")
flags.DEFINE_string("weights", "YOLO_small.ckpt", "yolo weights file")
flags.DEFINE_float('threshold', 0.2, "yolo")
flags.DEFINE_float('iou_threshold', 0.5, "yolo")
flags.DEFINE_string("job_name", "","One of 'ps' or 'worker'")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "List of hostname:port for ps jobs."
                    "This string should be the same on every host!!")
flags.DEFINE_string("worker_hosts", "localhost:3333",
                    "List of hostname:port for worker jobs."
                    "This string should be the same on every host!!")
flags.DEFINE_integer("task_index", None,
                     "Ps task index or worker task index, should be >= 0. task_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_integer("max_iter", None,
                     "Number of (global) training iterations to perform")
flags.DEFINE_integer("batch_size", 64, "Training batch size")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_boolean("sync_replicas", True,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean("allow_soft_placement", True, "True: allow")
flags.DEFINE_boolean("log_device_placement", False, "True: allow")
flags.DEFINE_string("gpu", "", "Which GPU to use, keep default if no GPU is used.")
flags.DEFINE_string("server_protocol", "grpc", "protocol for servers")
flags.DEFINE_enum('optimizer', 'sgd', ('momentum', 'sgd', 'rmsprop'),
                  'Optimizer to use: momentum or sgd or rmsprop')
flags.DEFINE_float('momentum', 0.9, 'Momentum for training.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum in RMSProp.')
flags.DEFINE_float('rmsprop_epsilon', 1.0, 'Epsilon term for RMSProp.')

FLAGS = flags.FLAGS    


def update_config_paths(data_dir, weights_file):

    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)  
    

def print_parameter_info():

    parameter_count = 0
    print("="*64)
    for var in tf.trainable_variables():
        print(var.name, end='\t')
        print("@%s" % var.device, end='\t')
        print(var.shape)
        parameter_count = parameter_count + reduce(lambda x, y: x * y, var.get_shape().as_list())
    print("Total parameter : %d, i.e., %.0f MB" % (parameter_count, parameter_count/1024/256))
    print("="*64)

def save_cfg(output_dir='.'):

    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        cfg_dict = cfg.__dict__
        for key in sorted(cfg_dict.keys()):
            if key[0].isupper():
                cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                f.write(cfg_str)

def get_optimizer(learning_rate):

    if FLAGS.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate,
                                    FLAGS.momentum, 
                                    use_nesterov=True)
    elif FLAGS.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate,
                                    FLAGS.rmsprop_decay,
                                    momentum=FLAGS.rmsprop_momentum,
                                    epsilon=FLAGS.rmsprop_epsilon)
    else:
        raise ValueError('Optimizer "%s" was not recognized',
                         FLAGS.optimizer)
    return opt

def build_model(batch_size):

    return YOLONet(batch_size=batch_size)

def load_data(batch_size):

    return pascal_voc('train', batch_size=batch_size)

def train(params):

    model = build_model(FLAGS.batch_size)
    print_parameter_info()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    ckpt_file = os.path.join(params['output_dir'], 'yolo')
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(params['output_dir'], flush_secs=60)

    global_step = tf.train.create_global_step()
    learning_rate = tf.train.exponential_decay(params['initial_learning_rate'], 
                                            global_step, 
                                            params['decay_steps'],
                                            params['decay_rate'], 
                                            params['staircase'], 
                                            name='learning_rate')
    optimizer = get_optimizer(learning_rate)
    train_op = slim.learning.create_train_op(
        model.total_loss, optimizer, global_step=global_step)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(),
                            allow_soft_placement=FLAGS.allow_soft_placement,
                            log_device_placement=FLAGS.log_device_placement)
    with tf.Session(config=config) as sess :
        sess.run(tf.global_variables_initializer())
        print("Session initialization complete.")

        if params['weights_file'] is not None:
            print('Restoring weights from: ' + params['weights_file'])
            saver.restore(sess, params['weights_file'])

        writer.add_graph(sess.graph)

        print('Start training ...')
        
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, params['max_iter'] + 1):

            load_timer.tic()
            images, labels = params['data'].get()
            load_timer.toc()
            feed_dict = {model.images: images,
                         model.labels: labels}

            if step % params['summary_iter'] == 0:
                train_timer.tic()
                summary_str, loss,  _ = sess.run(
                    [summary_op, model.total_loss, train_op],
                    feed_dict=feed_dict)
                train_timer.toc()

                log_str = "{} Epoch: {}, Step: {}, Learning rate: {}, Loss: {:5.3f}, Speed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}".format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    params['data'].epoch,
                    int(step),
                    round(learning_rate.eval(session=sess), 6),
                    loss,
                    train_timer.average_time,
                    load_timer.average_time,
                    train_timer.remain(step, params['max_iter']))
                print(log_str)

                writer.add_summary(summary_str, step)                
            else:
                train_timer.tic()
                sess.run(train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % params['save_iter'] == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    params['output_dir']))
                saver.save(sess, ckpt_file, global_step=global_step)

        print('Done training.')

def distributed_train(params):

    ps_hosts = re.findall(r'[\w\.:]+', FLAGS.ps_hosts)
    num_parameter_servers = len(ps_hosts)
    worker_hosts = re.findall(r'[\w\.:]+', FLAGS.worker_hosts)
    num_workers = len(worker_hosts)
    server = tf.train.Server({"ps":ps_hosts,"worker":worker_hosts}, 
        job_name=FLAGS.job_name, 
        task_index=FLAGS.task_index, 
        protocol=FLAGS.server_protocol)

    if FLAGS.job_name == "ps":
        server.join()
    else:
        is_chief = (FLAGS.task_index == 0)

    cluster_spec = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
    worker_prefix = '/job:worker/replica:0/task:%s' % FLAGS.task_index                 
    device_setter = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=worker_prefix)

    with tf.device(device_setter) :
        model = build_model(FLAGS.batch_size)
        print_parameter_info()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        ckpt_file = os.path.join(params['output_dir'], 'yolo')
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(params['output_dir'], flush_secs=60)

        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(params['initial_learning_rate'], 
                                                global_step, 
                                                params['decay_steps'],
                                                params['decay_rate'], 
                                                params['staircase'], 
                                                name='learning_rate')
        optimizer = get_optimizer(learning_rate)
        if FLAGS.sync_replicas :
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, 
                                                    replicas_to_aggregate=num_workers, 
                                                    total_num_replicas=num_workers)
        train_op = slim.learning.create_train_op(
            model.total_loss, optimizer, global_step=global_step)

        if is_chief:
            chief_queue_runner = optimizer.get_chief_queue_runner()
            init_tokens_op = optimizer.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 init_op=init_op,
                                 recovery_wait_secs=1,
                                 global_step=global_step)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(),
                                allow_soft_placement=FLAGS.allow_soft_placement,
                                log_device_placement=FLAGS.log_device_placement)
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
        sess = sv.prepare_or_wait_for_session(server.target, config=config)
        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        if is_chief:
            print("Starting chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        if params['weights_file'] is not None:
            print('Restoring weights from: ' + params['weights_file'])
            saver.restore(sess, params['weights_file'])

        writer.add_graph(sess.graph)

        print('Start distributed training ...')
        
        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, params['max_iter'] + 1):

            load_timer.tic()
            images, labels = params['data'].get()
            load_timer.toc()
            feed_dict = {model.images: images,
                         model.labels: labels}

            if step % params['summary_iter'] == 0:
                train_timer.tic()
                summary_str, loss, _ = sess.run(
                    [summary_op, model.total_loss, train_op],
                    feed_dict=feed_dict)
                train_timer.toc()

                log_str = "{} Epoch: {}, Step: {}, Learning rate: {}, Loss: {:5.3f}, Speed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}".format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    params['data'].epoch,
                    int(step),
                    round(learning_rate.eval(session=sess), 6),
                    loss,
                    train_timer.average_time,
                    load_timer.average_time,
                    train_timer.remain(step, params['max_iter']))
                print(log_str)

                writer.add_summary(summary_str, step)                
            else:
                train_timer.tic()
                sess.run(train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % params['save_iter'] == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    params['output_dir']))
                saver.save(sess, ckpt_file, global_step=global_step)

        print('Done distributed training.')


def main():

    if FLAGS.data_dir != cfg.DATA_PATH:
        update_config_paths(FLAGS.data_dir, FLAGS.weights)

    params = {}
    params['data'] = load_data(FLAGS.batch_size)
    params['weights_file'] = cfg.WEIGHTS_FILE
    if FLAGS.max_iter :
        params['max_iter'] = FLAGS.max_iter
    else :
        params['max_iter'] = cfg.MAX_ITER
    params['initial_learning_rate'] = cfg.LEARNING_RATE
    params['decay_steps'] = cfg.DECAY_STEPS
    params['decay_rate'] = cfg.DECAY_RATE
    params['staircase'] = cfg.STAIRCASE
    params['summary_iter'] = cfg.SUMMARY_ITER
    params['save_iter'] = cfg.SAVE_ITER
    params['output_dir'] = os.path.join(
        cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])
    save_cfg(params['output_dir'])

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    if FLAGS.job_name :
        distributed_train(params)
    else :
        train(params)

if __name__ == '__main__':
    main()
